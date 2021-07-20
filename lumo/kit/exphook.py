import os
import stat
import sys
from lumo.utils.exithook import wrap_before, wrap_after
from lumo.proc.path import libhome, local_dir
from .experiment import Experiment
from ..proc.const import EXP_CONST, FN


class ExpHook():
    def regist(self, exp: Experiment): self.exp = exp

    def on_start(self, exp: Experiment): pass

    def on_end(self, exp: Experiment): pass


class LastCmd(ExpHook):
    def on_start(self, exp: Experiment):
        with open('lastcmd.sh', 'w', encoding='utf-8') as w:
            w.write(' '.join(exp.exec_argv))

        st = os.stat('lastcmd.sh')
        os.chmod('lastcmd.sh', st.st_mode | stat.S_IEXEC)


class LogCmd(ExpHook):
    """a './cache/cmds.log' file will be generated, """

    def on_start(self, exp: Experiment):
        from lumo.proc.date import strftime
        fn = exp.project_cache_fn(f'{strftime("%y-%m-%d")}.log', 'cmds')
        res = exp.exec_argv

        with open(fn, 'a', encoding='utf-8') as w:
            w.write(f'{strftime("%H:%M:%S")}, {exp.test_root}, {res[0]}, {exp.commit_hash}\n')
            res[0] = os.path.basename(res[0])
            w.write(f"> {' '.join(res)}")
            w.write('\n\n')


class LogTestGlobally(ExpHook):
    def on_start(self, exp: Experiment):
        fn = os.path.join(libhome(), FN.TESTLOG)
        with open(fn, 'a', encoding='utf-8') as w:
            w.write(f'{exp.test_root}\n')


class LogTestLocally(ExpHook):
    def on_start(self, exp: Experiment):
        local_ = local_dir()
        if local_ is None:
            return
        fn = os.path.join(local_, FN.TESTLOG)
        with open(fn, 'a', encoding='utf-8') as w:
            w.write(f'{exp.test_root}\n')


class RegistRepo(ExpHook):
    def on_start(self, exp: Experiment):
        from ..proc.const import FN
        from ..proc.const import CFG
        from lumo.utils import safe_io as io
        fn = os.path.join(libhome(), FN.REPOSJS)
        res = None
        if os.path.exists(fn):
            res = io.load_json(fn)
        if res is None:
            res = {}

        inner = res.setdefault(exp.project_hash, {})
        inner['name'] = exp.project_name
        repos = inner.setdefault('repo', [])
        if exp.project_root not in repos:
            repos.append(exp.project_root)
        storages = inner.setdefault('exp_root', [])
        if exp.exp_root not in storages:
            storages.append(exp.exp_root)

        io.dump_json(res, fn)


class RecordAbort(ExpHook):
    def __init__(self):
        wrap_before(self.exc_end)

    def exc_end(self, exc_type, exc_val, exc_tb):
        import traceback
        res = traceback.format_exception(exc_type, exc_val, exc_tb)
        res = [i for i in res if 'in _newfunc' not in i]
        self.exp.writeline('exception', "".join(res))
        self.exp.end(
            end_code=1,
            exc_type=traceback.format_exception_only(exc_type, exc_val)[-1].strip()
        )


class PrintExpId(ExpHook):

    def regist(self, exp: Experiment):
        super().regist(exp)
        self.exp.add_exit_hook(self.on_exit)

    def on_exit(self, *args, **kwargs):
        print(f"END TEST {self.exp.short_uuid}")


class LogCMDAndTest(ExpHook):
    def on_start(self, exp: Experiment):
        from lumo.kit.logger import get_global_logger
        # get_global_logger().raw(f"{exp.test_root} | {' '.join(sys.argv)}")

    def on_end(self, exp: Experiment):
        from lumo.kit.logger import get_global_logger
        get_global_logger().raw(f"{exp.test_root} | {' '.join(sys.argv)}")
