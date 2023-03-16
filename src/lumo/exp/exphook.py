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
from lumo.proc.config import glob
from lumo.utils import safe_io as io
from lumo.utils.exithook import wrap_before
from lumo.utils.fmt import strftime, indent_print
from . import Experiment
from .base import BaseExpHook as BaseExpHook


class ExpHook(BaseExpHook):
    """A base class of hook for experiments that can be registered with an experiment."""

    def regist(self, exp: Experiment):
        self.exp = exp

    def on_start(self, exp: Experiment, *args, **kwargs): pass

    def on_end(self, exp: Experiment, end_code=0, *args, **kwargs): pass

    def on_progress(self, exp: Experiment, step, *args, **kwargs): pass

    def on_newpath(self, exp: Experiment, *args, **kwargs): pass


class LastCmd(ExpHook):
    """A hook to save the last command executed in an experiment.

    This hook saves the last command executed in an experiment to a shell script file in a specified directory. The saved
    file can be used to re-run the experiment with the same command.
    """
    configs = {'HOOK_LASTCMD_DIR': os.getcwd()}

    def on_start(self, exp: Experiment, *args, **kwargs):
        argv = exp.exec_argv
        fn = os.path.join(
            glob.get('HOOK_LASTCMD_DIR', self.configs['HOOK_LASTCMD_DIR']),
            f'run_{os.path.basename(argv[1])}.sh')

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


# class PathRecord(ExpHook):
#
#     def on_newpath(self, exp: Experiment, *args, **kwargs):
#         super().on_newpath(exp, *args, **kwargs)


class Diary(ExpHook):
    """A hook for logging experiment information to a diary file."""

    def on_start(self, exp: Experiment, *args, **kwargs):
        super().on_start(exp, *args, **kwargs)
        # with open(exp.root_file(f'{strftime("%y%m%d")}.log', 'diary'), 'a') as w:
        #     w.write(f'{strftime("%H:%M:%S")}, {exp.test_root}\n')


class RecordAbort(ExpHook):
    """A hook to record and handle experiment aborts.
    """

    def regist(self, exp: Experiment):
        super().regist(exp)
        wrap_before(self.exc_end)

    def exc_end(self, exc_type, exc_val, exc_tb):
        import traceback
        res = traceback.format_exception(exc_type, exc_val, exc_tb)
        res = [i for i in res if 'in _newfunc' not in i]

        self.exp.dump_info('exception', {
            'exception_type': exc_type.__name__,
            'exception_content': "".join(res)
        })

        self.exp.end(end_code=1)


# class TimeMonitor(ExpHook):
#     def _create_agent(self, exp: Experiment):
#         from lumo.exp import agent
#         cmd = [
#             sys.executable, '-m', agent.__spec__.name,
#             f"--state_key=state",
#             f"--pid={os.getpid()}",
#             f"--exp_name={exp.exp_name}",
#             f"--test_name={exp.test_name}",
#             f"--test_root={exp.test_root}",
#             # f"--params={sys.argv}"
#         ]
#         subprocess.Popen(' '.join(cmd),
#                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
#                          start_new_session=True, cwd=os.getcwd(), env=os.environ)
#
#     def on_start(self, exp: Experiment, *args, **kwargs):
#         super().on_start(exp)
#         self._create_agent(exp)
#         exp.dump_info('state', {
#             'start': strftime(),
#             'end': strftime()
#         })


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
        import inspect

        if not git_enable():
            exp.dump_info('git', {
                'code': -1,
                'msg': 'git not found'
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

        dep_hash = hash(dep_source)
        commit_ = git_commit(key='lumo', info=exp.info_dir, filter_files=filter_files)

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
        file = exp.mk_rpath('repos', hash(exp.project_root))
        exps = {}
        if os.path.exists(file):
            exps = io.load_json(file)
        res = exps.setdefault(exp.project_root, list())
        if exp.exp_dir not in res:
            res.append(exp.exp_dir)
        io.dump_json(exps, file)


class LockFile(ExpHook):
    """A class for locking dependencies for an experiment.
    Locks the specified dependencies for the experiment and saves them to a file.
    """

    def on_start(self, exp: Experiment, *args, **kwargs):
        basic = get_lock('lumo',
                         'joblib',
                         'fire',
                         'psutil',
                         'hydra',
                         'omegaconf',
                         'decorator',
                         'numpy',
                         'torch',
                         )
        if basic['torch'] is not None:
            import torch
            if torch.cuda.is_available():
                basic['torch.version.cuda'] = torch.version.cuda

        exp.dump_info('lock', basic)


class FinalReport(ExpHook):
    """A class for generating a final report for an experiment.

      Prints the experiment's properties, tags, paths, and execute command.
      """

    def on_end(self, exp: Experiment, end_code=0, *args, **kwargs):
        # if end_code == 0:
        print('-----------------------------------')
        # print('')

        print('Properties:')
        indent_print(pformat(exp.properties))
        print('Execute:')
        indent_print(' '.join(exp.exec_argv))
        print('-----------------------------------')
