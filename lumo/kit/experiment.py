"""
Experiment
    - 负责记录实验中产生的所有信息
    - 管理实验路径


"""
import os
import time
import random
import sys
import traceback
from typing import TYPE_CHECKING
from lumo.base_classes import attr

from lumo.utils.filebranch import FileBranch
from lumo.utils.safe_io import IO
from lumo.proc.dist import is_dist, is_main
from lumo.proc.path import local_dir, libhome

if TYPE_CHECKING:
    from .exphook import ExpHook


def listener(func):
    def inner(self: Experiment, *args, **kwargs):
        return func(*args, **kwargs)

    return inner


class Experiment:
    """
    每一个经由 experiment 辅助的实验，都会在以下几个位置存有记录，方便回溯：
     - ~/.lumo/
     - <working_dir>/.lumo/
     - <save_dir>/ # 记录相对路径

    TODO
    主要功能：
     - 以实验为单位，维护在该实验上做的全部试验，用于提供该次"试验唯一"的目录，该"实验唯一"的目录，"用户唯一"的目录
     - 以试验为单位，存储试验过程中提交到 Experiment 的全部信息（配置信息等）

    主要功能：
     - 以试验为单位，在试验开始前提交本次运行快照

    其中，存储目录从用户配置（本地）读取，但也可以优先通过环境变量控制

    根目录 -> 实验目录 -> 试验目录 -> 试验目录下子目录

    使用方法：

    with Experiment(exp_name) as exp:

    """

    def __init__(self, exp_name, test_name=None, root=None):
        if root is None:
            root = libhome()
        self._hooks = {}
        self._exp_name = exp_name
        self._test_name = test_name

        self._path_mem = set()

        def foo(path):
            if path not in self._path_mem:
                self._path_mem.add(path)
                for v in self._hooks.values():  # type:ExpHook
                    v.on_newpath(self)

        self._tree = FileBranch(root, listener=foo)
        self.add_exit_hook(self._auto_end)

    def _auto_end(self, *args):
        self.end(end_code=0)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        extra = {}
        if exc_type is not None:
            exc_type = traceback.format_exception_only(exc_type, exc_val)[-1].strip()
            extra['exc_type'] = exc_type
            extra['end_info'] = str(exc_type)
            extra['end_code'] = 1
            extra['exc_stack'] = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
        self.end(**extra)

    def _create_test_name(self):
        from lumo.proc.date import strftime, timehash
        fs = os.listdir(self.exp_root)
        date_str = strftime('%y%m%d')
        fs = [i for i in fs if i.startswith(date_str)]
        _test_name = f"{date_str}.{len(fs):03d}.{timehash()[-6:-4]}t"
        return _test_name

    def test_file(self, name, *dirs):
        return self.test_branch.file(name, *dirs)

    def test_dir(self, *dirs):
        return self.test_branch.makedir(*dirs)

    def exp_file(self, name, *dirs):
        return self.exp_branch.file(name, *dirs)

    def exp_dir(self, *dirs):
        return self.exp_branch.makedir(*dirs)

    def blob_file(self, name, *dirs):
        return self.blob_branch.file(name, *dirs)

    def blob_dir(self, *dirs):
        return self.blob_branch.makedir(*dirs)

    def backup_file(self, name, *dirs):
        return self.backup_branch.file(name, *dirs)

    def backup_dir(self, *dirs):
        return self.backup_branch.makedir(*dirs)

    def cache_file(self, name, *dirs):
        return self.cache_branch.file(name, *dirs)

    def cache_dir(self, *dirs):
        return self.cache_branch.makedir(*dirs)

    def load_info(self, key):
        fn = self.test_file(f'{key}.json', 'info')
        if not os.path.exists(fn):
            return {}
        return attr.from_dict(IO.load_json(fn))

    def dump_info(self, key, info: dict, append=False):
        info = attr(info).jsonify()
        fn = self.test_file(f'{key}.json', 'info')
        if append:
            old_info = self.load_info(key)
            old_info.update(info)
            info = old_info
        IO.dump_json(info, fn)
        return fn

    def dump_string(self, fn, info):
        fn = self.test_file(fn, 'info')
        IO.dump_text(str(info), fn)
        return fn

    def add_tag(self, tag):
        fn = self.test_file(tag, 'tag')
        with open(fn, 'w'):
            pass
        return fn

    def start(self):
        for hook in self._hooks.values():  # type: ExpHook
            hook.on_start(self)
        return self

    def end(self, end_code=0, *args, **extra):
        for hook in self._hooks.values():  # type: ExpHook
            hook.on_end(self, end_code=end_code, *args, **extra)
        return self

    def update(self, step, *args, **kwargs):
        for hook in self._hooks.values():  # type: ExpHook
            hook.on_progress(self, step, *args, **kwargs)

    def set_hook(self, hook: 'ExpHook'):
        hook.regist(self)
        self._hooks[hook.__class__.__name__] = hook
        return self

    def add_exit_hook(self, func):
        import atexit
        def exp_func():
            func(self)

        atexit.register(exp_func)

    @property
    def paths(self):
        return sorted(self._path_mem)

    @property
    def root_branch(self):
        return self._tree

    @property
    def exp_branch(self):
        return self.root_branch.branch('experiment', self.exp_name)

    @property
    def blob_branch(self):
        return self.root_branch.branch('blob', self.exp_name, self.test_name)

    @property
    def backup_branch(self):
        return self.root_branch.branch('backup', self.exp_name)

    @property
    def cache_branch(self):
        return self.root_branch.parent().branch('cache')

    @property
    def test_branch(self):
        return self.exp_branch.branch(self.test_name)

    @property
    def project_branch(self):
        return FileBranch(self.project_root)

    @property
    def exec_argv(self):
        return [os.path.basename(sys.executable), *sys.argv]

    @property
    def project_name(self):
        return os.path.basename(self.project_root)

    @property
    def exp_name(self):
        return self._exp_name

    @property
    def test_name(self):
        """Create unique name for the current test"""
        if self._test_name is None:
            if is_dist():  # 分布式非主线程等待以获取 test_name
                flag_fn = f'.{os.getppid()}'
                if is_main() > 0:
                    time.sleep(random.randint(2, 4))
                    fn = self.exp_file(flag_fn)
                    if os.path.exists(fn):
                        with open(fn, 'r') as r:
                            self._test_name = r.readline().strip()
                else:
                    self._test_name = self._create_test_name()
                    fn = self.exp_file(flag_fn)
                    with open(fn, 'w') as w:
                        w.write(self._test_name)
            else:
                self._test_name = self._create_test_name()

        return self._test_name

    @property
    def project_root(self):
        return local_dir()

    @property
    def exp_root(self) -> str:
        """root dir for current experiment"""
        return self.exp_branch.root

    @property
    def test_root(self) -> str:
        """Root dir for current test"""
        return self.test_branch.root


class SimpleExperiment(Experiment):

    def __init__(self, exp_name, test_name=None, root=None):
        super().__init__(exp_name, test_name, root)
        from . import exphook
        self.set_hook(exphook.LastCmd())
        self.set_hook(exphook.LogCMDAndTest())
        self.set_hook(exphook.LockFile())
        self.set_hook(exphook.ExecuteInfo())
        self.set_hook(exphook.GitCommit())
        self.set_hook(exphook.RecordAbort())
        self.set_hook(exphook.Diary())
        self.set_hook(exphook.BlobPath())
        self.set_hook(exphook.TimeMonitor())


class TrainerExperiment(SimpleExperiment):

    def __init__(self, exp_name, test_name=None):
        super().__init__(exp_name, test_name)

    class VAR_KEY:
        WRITER = 'board'

    @property
    def log_dir(self):
        return self.test_root

    @property
    def params_fn(self):
        res = self.test_file('params.json')
        return res

    @property
    def board_args(self):
        log_dir = self.test_dir('board')
        return {
            'filename_suffix': '.bd',
            'log_dir': log_dir,
        }

    @property
    def saver_dir(self):
        res = self.blob_dir('saver')
        return res

    def dump_train_info(self, epoch):
        self.dump_info('trainer', {
            'epoch': epoch
        }, append=True)
