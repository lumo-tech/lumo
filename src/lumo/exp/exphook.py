import os
import stat
import subprocess
import sys
from collections import OrderedDict
from pprint import pformat

from joblib import hash

from lumo.decorators.lazy_required import is_lib_available
from lumo.proc.dependency import get_lock
from lumo.proc.dist import is_main
from lumo.utils import safe_io as io
from lumo.utils.exithook import wrap_before
from lumo.utils.fmt import strftime, indent_print
from . import Experiment
from .base import ExpHook as BaseExpHook


class ExpHook(BaseExpHook):
    def regist(self, exp: Experiment):
        self.exp = exp

    def on_start(self, exp: Experiment, *args, **kwargs): pass

    def on_end(self, exp: Experiment, end_code=0, *args, **kwargs): pass

    def on_progress(self, exp: Experiment, step, *args, **kwargs): pass

    def on_newpath(self, exp: Experiment, *args, **kwargs): pass


class LastCmd(ExpHook):
    def on_start(self, exp: Experiment, *args, **kwargs):
        argv = exp.exec_argv
        fn = f'run_{os.path.basename(argv[1])}.sh'

        strings = OrderedDict.fromkeys([i.lstrip('#').strip() for i in io.load_text(fn).split('\n')])

        cur = f"{' '.join(argv)} $@"
        if cur in strings:
            strings.pop(cur)

        with open(fn, 'w', encoding='utf-8') as w:
            w.write('\n'.join([f'# {i}' for i in strings.keys()]))
            w.write('\n\n')
            w.write(cur)

        st = os.stat(fn)
        os.chmod(fn, st.st_mode | stat.S_IEXEC)


class PathRecord(ExpHook):

    def on_newpath(self, exp: Experiment, *args, **kwargs):
        super().on_newpath(exp, *args, **kwargs)


class Diary(ExpHook):
    def on_start(self, exp: Experiment, *args, **kwargs):
        super().on_start(exp, *args, **kwargs)
        with open(exp.root_file(f'{strftime("%y%m%d")}.log', 'diary'), 'a') as w:
            w.write(f'{strftime("%H:%M:%S")}, {exp.test_root}\n')


class RecordAbort(ExpHook):
    def regist(self, exp: Experiment):
        super().regist(exp)
        wrap_before(self.exc_end)

    def exc_end(self, exc_type, exc_val, exc_tb):
        import traceback
        res = traceback.format_exception(exc_type, exc_val, exc_tb)
        res = [i for i in res if 'in _newfunc' not in i]
        self.exp.dump_string('exception', "".join(res))
        self.exp.end(
            end_code=1,
            exc_type=traceback.format_exception_only(exc_type, exc_val)[-1].strip()
        )


class TimeMonitor(ExpHook):
    def _create_agent(self, exp: Experiment):
        from lumo.exp import agent
        cmd = [
            sys.executable, '-m', agent.__spec__.name,
            f"--state_key=state",
            f"--pid={os.getpid()}",
            f"--exp_name={exp.exp_name}",
            f"--test_name={exp.test_name}",
            f"--test_root={exp.test_root}",
            # f"--params={sys.argv}" # TODO add sys.argv
        ]
        subprocess.Popen(' '.join(cmd),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                         start_new_session=True)

    def on_start(self, exp: Experiment, *args, **kwargs):
        super().on_start(exp)
        self._create_agent(exp)
        exp.dump_info('state', {
            'start': strftime(),
            'end': strftime()
        })


class GitCommit(ExpHook):

    def on_start(self, exp: Experiment, *args, **kwargs):
        if not is_lib_available('git'):
            exp.dump_info('git', {
                'code': -1,
                'msg': 'git not installed'
            })
            return

        from lumo.utils.repository import git_enable, git_commit, git_dir
        from lumo.utils.ast import analyse_module_dependency
        from lumo.utils.hash import hash_iter
        import inspect

        if not git_enable():
            exp.dump_info('git', {
                'code': -1,
                'msg': 'git disabled'
            })
            return

        if not is_main():
            return

        import __main__

        root = git_dir()
        mem = analyse_module_dependency(__main__, root=root)

        filter_files = set()
        dep_source = []
        for fn, dep_module in sorted(mem.items()):
            if os.path.commonprefix([fn, root]) == root:
                filter_files.add(os.path.relpath(fn, root))
                try:
                    dep_source.append(inspect.getsource(dep_module))
                except OSError:
                    pass

        dep_hash = hash_iter(*dep_source)
        commit_ = git_commit(key='lumo', info=exp.test_root, filter_files=filter_files)

        if commit_ is None:
            exp.dump_info('git', {
                'code': -2,
                'msg': 'commit error'
            })
            return

        commit_hex = commit_.hexsha[:8]
        exp.dump_info('git', {
            'commit': commit_hex,
            'repo': exp.project_root,
            'dep_hash': dep_hash,
        })

        file = exp.root_file(hash(exp.project_root), 'repos')
        exps = {}
        if os.path.exists(file):
            exps = io.load_json(file)
        res = exps.setdefault(exp.project_root, list())
        if exp.exp_root not in res:
            res.append(exp.exp_root)
        io.dump_json(exps, file)


class LockFile(ExpHook):

    def on_start(self, exp: Experiment, *args, **kwargs):
        exp.dump_info('lock', get_lock('torch', 'numpy',
                                       'joblib',
                                       'psutil',
                                       'decorator',
                                       'torch',
                                       'numpy',
                                       'accelerate',
                                       'hydra',
                                       'omegaconf', ))


class FinalReport(ExpHook):
    def on_end(self, exp: Experiment, end_code=0, *args, **kwargs):
        # if end_code == 0:
        print('-----------------------------------')
        # print('')

        print('Properties:')
        indent_print(pformat(exp.properties))
        print('Tags:')
        indent_print(pformat(exp.tags))
        print('Use paths:')
        indent_print(pformat(exp.paths))
        print('Execute:')
        indent_print(' '.join(exp.exec_argv))
        print('-----------------------------------')
