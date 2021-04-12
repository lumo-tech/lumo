import os
import pickle
import sys
import traceback
from typing import Union, TYPE_CHECKING
from uuid import uuid1
from lumo.base_classes import attr
from lumo.kit.environ import globs
from lumo.utils import paths
from lumo.utils import safe_io as io
from lumo.utils.dates import strftime
from lumo.utils.keys import CFG, EXP, FN, LIBRARY_NAME
from lumo.utils.repository import commit as git_commit

PATH_DEFAULT = CFG.PATH.DEFAULT

if TYPE_CHECKING:
    from .exphook import ExpHook


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

    def __init__(self, exp_name, test_name=None):
        self._hooks = []
        self._exp_name = exp_name
        self._test_name = test_name

        from .exphook import RegistRepo, RecordAbort, LogTestLocally, LogTestGlobally
        (
            self.add_hook(RegistRepo())
                .add_hook(RecordAbort())
                .add_hook(LogTestLocally())
                .add_hook(LogTestGlobally())
        )

    def _create_dir(self, root, dirname):
        res = paths.checkpath(root, dirname)
        return res

    def _create_fn(self, basename, dirname, root):
        if dirname is not None:
            res = paths.checkpath(root, dirname)
        else:
            res = root
        return os.path.join(res, basename)

    def _create_agent(self):
        import subprocess, sys
        from lumo.kit import agent
        cmd = [
            sys.executable, '-m', agent.__spec__.name,
            f"--pid={os.getpid()}",
            f"--test_name={self.test_name}",
            f"--exp_name={self.exp_name}",
        ]
        subprocess.Popen(' '.join(cmd),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                         start_new_session=True)

    def cache_dir(self, dirname):
        return self._create_dir(self.cache_root, dirname)

    def project_cache_fn(self, name, dirname=None):
        return self._create_fn(name, dirname, self.project_cache_root)

    def project_cache_dir(self, dirname):
        return self._create_dir(self.project_cache_root, dirname)

    def exp_dir(self, name):
        return self._create_dir(self.exp_root, name)

    def project_dir(self, name):
        return self._create_dir(self.project_root, name)

    def project_fn(self, basename, dirname=None):
        return self._create_fn(basename, dirname, self.project_root)

    def exp_fn(self, basename, dirname=None):
        return self._create_fn(basename, dirname, self.exp_root)

    def test_dir(self, name):
        return self._create_dir(self.test_root, name)

    def test_fn(self, basename, dirname=None):
        return self._create_fn(basename, dirname, self.test_root)

    def load_info(self, key):
        fn = self.test_fn(self._create_info_basename(key), FN.D_JSON)
        if not os.path.exists(fn):
            return None
        return attr.from_dict(io.load_json(fn))

    def dump_info(self, key, info: dict, append=False):
        info = attr(info).jsonify()
        fn = self.test_fn(self._create_info_basename(key), FN.D_JSON)
        if append and os.path.exists(fn):
            old_info = self.load_info(key)
            for k, v in info.items():
                old_info[k] = v
            info = old_info
        io.dump_json(info, fn)
        return fn

    def load_pkl(self, key):
        fn = self.test_fn(self._create_bin_basename(key), FN.D_PKL)
        if not os.path.exists(fn):
            return None
        with open(fn, 'r') as r:
            return pickle.load(r)

    def dump_pkl(self, key, info):
        fn = self.test_fn(self._create_bin_basename(key), FN.D_PKL)
        with open(fn, 'wb') as w:
            pickle.dump(info, w)
        return fn

    def writeline(self, key, value: str):
        fn = self.test_fn(f"{key}.txt", FN.D_LINE)
        with open(fn, 'w', encoding='utf-8') as w:
            w.write(value)

    def readline(self, key):
        fn = self.test_fn(f"{key}.txt", FN.D_LINE)
        if not os.path.exists(fn):
            return ''
        with open(fn, 'r', encoding='utf-8') as r:
            return ''.join(r.readlines())

    def readlines(self, raw=False) -> Union[dict, str]:
        line_root = os.path.join(self.test_root, FN.D_LINE)
        fs_ = os.listdir(line_root)
        fs_ = [f for f in fs_ if f.endswith('.txt')]
        res = {}
        for f in fs_:
            key = f[:-1]
            line = self.readline(key)
            res[key] = line
        if raw:
            return res
        else:
            strs = []
            for k, v in res.items():
                strs.append(k)
                strs.append('-' * len(k))
                strs.append(v)
                strs.append('=' * len(k))
            return '\n'.join(strs)

    def _create_bin_basename(self, key):
        return f"{key}.pkl"

    def _create_info_basename(self, key):
        return f"{key}.json"

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        extra = {}
        if exc_type is not None:
            exc_type = traceback.format_exception_only(exc_type, exc_val)[-1].strip()
            extra['exc_type'] = exc_type
            extra['end_info'] = str(exc_type)
            extra['end_code'] = 1
            self.writeline(EXP.EXCEPTION,
                           "".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
        self.end(**extra)

    @property
    def exec_argv(self):
        return [sys.executable, *sys.argv]

    @property
    def project_hash(self):
        hash_fn = os.path.join(paths.local_dir(), FN.PHASH)
        return io.load_string(hash_fn)

    @property
    def project_name(self):
        return os.path.basename(self.project_root)

    @property
    def exp_name(self):
        return self._exp_name

    @property
    def test_name(self):
        """Identity for current test"""
        if self._test_name is None:
            from lumo.utils.dates import strftime
            fs = os.listdir(self.exp_root)
            date_str = strftime('%y%m%d')
            fs = [i for i in fs if i.startswith(date_str)]
            self._test_name = f"{date_str}.{len(fs):03d}t"
        return self._test_name

    @property
    def storage_root(self):
        """experiments root dir, default is `~/.lumo/experiments`"""
        path = globs.get_first(CFG.PATH.LOCAL_EXP, CFG.PATH.GLOBAL_EXP,
                               default=PATH_DEFAULT.GLOBAL_EXP)

        return paths.checkpath(path)

    @property
    def cache_root(self) -> str:
        """cache root for lumo, default"""
        res = globs.get_first(CFG.PATH.CACHE,
                              default=PATH_DEFAULT.CACHE)
        return paths.checkpath(res)

    @property
    def exp_root(self) -> str:
        """root dir for current experiment"""
        return paths.checkpath(self.storage_root, self.exp_name)

    @property
    def test_root(self) -> str:
        """Root dir for current test"""
        return paths.checkpath(self.exp_root, self.test_name)

    @property
    def project_root(self) -> str:
        """git repository root(working dir)"""
        return paths.repo_dir()

    @property
    def project_cache_root(self) -> str:
        """<project_root>/.cache"""
        return paths.checkpath(self.project_root, PATH_DEFAULT.LOCAL_CACHE)

    @property
    def commit_hash(self) -> str:
        from lumo.utils.repository import _commits_map
        res = _commits_map.get(LIBRARY_NAME, None)
        if res is not None:
            return res.hexsha[:8]
        return ''

    def dump_experiment_info(self):
        self.dump_info(EXP.STATE, {
            'start': strftime(),
            'end': strftime()
        })

        self.dump_info(EXP.EXECUTE, {
            CFG.PATH.REPO: self.project_root,
            CFG.PATH.CACHE: os.getcwd(),
            'exec_file': sys.argv[0],
            'exec_bin': sys.executable,
            'exec_argv': sys.argv
        })

        self.dump_info(EXP.PROJECT, {
            'hash': self.project_hash,
            'root': self.project_root,
        })

        no_commit = globs.get_first(CFG.STATE.DISABLE_GIT,
                                    default=CFG.STATE.DEFAULT.DISABLE_GIT)
        if not no_commit:
            commit_ = git_commit(key=LIBRARY_NAME, info=self.test_root).hexsha[:8]
            self.writeline('commit', commit_)
            self.dump_info(EXP.GIT, {
                'commit': commit_,
                CFG.PATH.REPO: self.project_root,
            })
        self.writeline('uuid', uuid1().hex)

        from lumo import __version__
        self.dump_info(EXP.VERSION, {
            'lumo': __version__,
        })

    def start(self):
        self._create_agent()
        self.dump_experiment_info()
        for hook in self._hooks:  # type: ExpHook
            hook.on_start(self)
        return self

    def end(self, enc_code=0, **extra):
        self.dump_info(EXP.STATE, {
            'end_code': enc_code,
            **extra,
        }, append=True)

        for hook in self._hooks:  # type: ExpHook
            hook.on_end(self)
        return self

    def add_hook(self, hook: 'ExpHook'):
        hook.regist(self)
        self._hooks.append(hook)
        return self

    def add_exit_hook(self, func):
        import atexit
        def exp_func():
            func(self)

        atexit.register(exp_func)


class TrainerExperiment(Experiment):

    def __init__(self, exp_name, test_name=None):
        super().__init__(exp_name, test_name)
        from .exphook import LastCmd, LogCmd
        self.add_hook(LastCmd())
        self.add_hook(LogCmd())

    class VAR_KEY:
        WRITER = 'board'

    @property
    def log_dir(self):
        return self.test_root

    @property
    def params_fn(self):
        # self.writeline('params',)
        return self.test_fn('params.yaml')

    @property
    def board_args(self):
        log_dir = self.test_dir('board')
        self.writeline('writer', log_dir)
        return {
            'filename_suffix': '.bd',
            'log_dir': log_dir,
        }

    @property
    def saver_dir(self):
        return self.test_dir('saver')

    @property
    def rnd_dir(self):
        return self.project_cache_dir('rnd')

    def dump_experiment_info(self):
        super().dump_experiment_info()
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.cuda_version
            else:
                cuda_version = '0'
            self.dump_info(EXP.VERSION, {
                'torch': torch.__version__,
                'cuda': cuda_version
            }, append=True)
        except:
            pass

    def dump_train_info(self, epoch):
        self.dump_info(EXP.TRAINER, {
            'epoch': epoch
        }, append=True)
