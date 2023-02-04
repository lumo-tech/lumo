import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Union
from lumo.utils.logger import Logger
from lumo.decorators.process import call_on_main_process_wrap
from lumo.proc.path import blobroot, exproot, cache_dir, libhome
from .base import ExpHook
from ..proc.dist import is_dist, is_main, local_rank
from ..proc.path import exproot, local_dir
from ..utils import safe_io as io
from ..utils.fmt import can_be_filename


def checkdir(path: Union[Path, str]):
    if isinstance(path, str):
        os.makedirs(path, exist_ok=True)
    elif isinstance(path, Path):
        path.mkdir(parents=True, exist_ok=True)
    return path


class Experiment:
    """
    (by default), the directory structure is as following:
    .lumo (libroot)
        - experiments # (exp_root) record information (e.g., .log, params files, etc.)
            - {experiment-name-1}
                - {test-1}
                - {test-2}
            - {experiment-name-2}
                - {test-1}
                - {test-2}
        - blob # (blob_root) record big binary files (like model state dicts)
            - {experiment-name-1}
                - {test-1}
                - {test-2}
            - {experiment-name-2}
                - {test-1}
                - {test-2}
        - metric # (metric_root) record metrics (by trainer.database)
            - {experiment-name-1}
                - {test-1}
                - {test-2}
            - {experiment-name-2}
                - {test-1}
                - {test-2}
    """

    def __init__(self, exp_name: str, root=None):
        if not can_be_filename(exp_name):
            raise ValueError(f'Experiment name should be a ligal filename(bettor only contain letter or underline),'
                             f'but got {exp_name}.')

        self._prop = {}
        self._prop['exp_name'] = exp_name
        self._hooks = {}
        if root is None:
            root = libhome()
        self._root = Path(os.path.abspath(root))
        self.add_exit_hook(self.end)
        self.logger = Logger()

    @property
    def exp_name(self):
        return self._prop['exp_name']

    @property
    def _test_name(self):
        return self._prop.get('test_name', None)

    @_test_name.setter
    def _test_name(self, value):
        self._prop['test_name'] = value

    @property
    def test_name_with_dist(self):
        """
        Create different test_name for each process in multiprocess training.
        Returns: in main process, will just return test_name itself,
        in subprocess,  "{test_name}.{local_rank()}"
        """
        raw = self.test_name
        if not is_dist():
            return raw

        if is_main():
            return raw
        else:
            return f"{raw}.{local_rank()}"

    @property
    def test_name(self):
        """Assign unique space(directory) for this test"""
        if self._test_name is None:
            if is_dist():  # if train in distribute mode, subprocess will wait a few seconds to wait main process.
                flag_fn = f'.{os.getppid()}'
                if is_main():
                    self._test_name = self._create_test_name()
                    fn = self.exp_file(flag_fn)
                    with open(fn, 'w') as w:
                        w.write(self._test_name)
                else:
                    time.sleep(random.randint(2, 4))
                    fn = self.exp_file(flag_fn)
                    if os.path.exists(fn):
                        with open(fn, 'r') as r:
                            self._test_name = r.readline().strip()
            else:
                self._test_name = self._create_test_name()

        return self._test_name

    def _create_test_name(self):
        from lumo.proc.date import timehash
        from ..utils.fmt import strftime
        fs = os.listdir(self.exp_root)
        date_str = strftime('%y%m%d')
        fs = [i for i in fs if i.startswith(date_str)]
        _test_name = f"{date_str}.{len(fs):03d}.{timehash()[-6:-4]}t"
        return _test_name

    @property
    def root_branch(self):
        val = self._root
        return checkdir(val)

    @property
    def lib_root(self):
        return self.root_branch.as_posix()

    @property
    def exp_branch(self):
        val = Path(exproot()).joinpath(self.exp_name)
        return checkdir(val)

    @property
    def blob_branch(self):
        val = Path(blobroot()).joinpath(self.exp_name, self.test_name)
        return checkdir(val)

    @property
    def test_branch(self):
        val = self.exp_branch.joinpath(self.test_name)
        return checkdir(val)

    def dump_info(self, key: str, info: dict, append=False, info_dir='info', set_prop=True):
        fn = self.test_file(f'{key}.json', info_dir)
        if append:
            old_info = self.load_info(key, info_dir=info_dir)
            old_info.update(info)
            info = old_info
        if set_prop:
            self.set_prop(key, info)
        io.dump_json(info, fn)

    def load_info(self, key: str, info_dir='info'):
        fn = self.test_file(f'{key}.json', info_dir)
        if not os.path.exists(fn):
            return {}
        return io.load_json(fn)

    def dump_string(self, key: str, info: str):
        fn = self.test_file(f'{key}.str', 'text')
        io.dump_text(info, fn)
        self.set_prop(key, info)

    def load_string(self, key: str):
        fn = self.test_file(f'{key}.str', 'text')
        if not os.path.exists(fn):
            return ''
        return io.load_text(fn)

    @property
    def tags(self):
        tags = {}
        for path in self.test_branch.joinpath('tags').glob('tag.*.json'):
            ptags = io.load_json(path.as_posix())  # type: dict
            tags.setdefault(path.suffixes[0].strip('.'), []).extend(ptags.keys())
        return tags

    def add_tag(self, tag: str, name_space: str = 'default'):
        self.dump_info(f'tag.{name_space}', {
            tag: None
        }, append=True, info_dir='tags', set_prop=False)

    def exp_file(self, filename, *args):
        """

        Args:
            filename:
            *args:
            mkdir:

        Returns:

        """
        parent = self.exp_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def test_file(self, filename, *args):
        parent = self.test_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def exp_dir(self, *args):
        """

        Args:
            filename:
            *args:
            mkdir:

        Returns:

        """
        parent = self.exp_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def root_file(self, filename, *args):
        parent = self.root_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def root_dir(self, *args):
        """

        Args:
            filename:
            *args:
            mkdir:

        Returns:

        """
        parent = self.root_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def test_dir(self, *args):
        parent = self.test_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def blob_file(self, filename, *args):
        parent = self.blob_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def blob_dir(self, *args):
        """

        Args:
            filename:
            *args:
            mkdir:

        Returns:

        """
        parent = self.blob_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        extra = {}
        if exc_type is not None:
            exc_type = traceback.format_exception_only(exc_type, exc_val)[-1].strip()
            extra['exc_type'] = exc_type
            extra['end_info'] = str(exc_type)
            extra['end_code'] = 1
            extra['exc_stack'] = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
        self.end(**extra)

    @call_on_main_process_wrap
    def add_exit_hook(self, func):
        import atexit
        def exp_func():
            func(self)

        atexit.register(exp_func)

    @call_on_main_process_wrap
    def initial(self):
        self.add_tag(self.__class__.__name__, 'exp_type')
        self.dump_info('execute', {
            'repo': self.project_root,
            'cwd': os.getcwd(),
            'exec_file': sys.argv[0],
            'exec_bin': sys.executable,
            'exec_argv': sys.argv
        })

    @call_on_main_process_wrap
    def start(self):
        if self.get_prop('start', False):
            return
        self.initial()
        self.set_prop('start', True)
        for hook in self._hooks.values():  # type: ExpHook
            hook.on_start(self)
        return self

    @call_on_main_process_wrap
    def end(self, end_code=0, *args, **extra):
        if not self.get_prop('start', False):
            return
        if self.get_prop('end', False):
            return
        self.set_prop('end', True)
        for hook in self._hooks.values():  # type: ExpHook
            hook.on_end(self, end_code=end_code, *args, **extra)
        return self

    @property
    def repo_name(self):
        """repository name"""
        return self.project_name

    @property
    def project_name(self):
        """same as repository name, directory name of project root"""
        return os.path.basename(self.project_root)

    @property
    def project_root(self):
        return local_dir()

    @property
    def exp_root(self):
        """path to multiple tests of this experiment"""
        return self.exp_branch.as_posix()

    @property
    def test_root(self):
        """path to record information of one experiment"""
        return self.test_branch.as_posix()

    @property
    def blob_root(self):
        """path to storing big binary files"""
        return self.blob_branch.as_posix()

    def __getitem__(self, item):
        return self._prop[item]

    def __setitem__(self, key, value):
        self._prop[key] = value

    def get_prop(self, key, default=None):
        return self._prop.get(key, default)

    def has_prop(self, key):
        return key in self._prop

    def set_prop(self, key, value):
        self._prop[key] = value

    @property
    def properties(self):
        return self._prop

    @property
    def paths(self) -> dict:
        return {
            'root': self.root_branch.as_posix(),
            'exp_root': self.exp_root,
            'test_root': self.test_root,
            'blob_root': self.blob_root,
        }

    @property
    def enable_properties(self) -> set:
        return set(self._prop.keys())

    @call_on_main_process_wrap
    def set_hook(self, hook: ExpHook):
        hook.regist(self)
        self.logger.info(f'Register {hook}.')
        self._hooks[hook.__class__.__name__] = hook
        self.add_tag(hook.__class__.__name__, 'hooks')
        return self

    def load_prop(self):
        for f in os.listdir(self.test_dir('info')):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_info(key))

        for f in os.listdir(self.test_dir('text')):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_string(key))

    @classmethod
    def from_disk(cls, path):
        from .finder import is_test_root
        if not is_test_root(path):
            raise ValueError(f'{path} is not a valid test_root')

        test_root = Path(path)
        root = test_root.parent.parent.parent.as_posix()
        self = cls(test_root.parent.name, root=root)
        self._test_name = test_root.name
        self.load_prop()
        return self

    @property
    def exec_argv(self):
        execute_info = self.get_prop('execute')
        try:
            return [os.path.basename(execute_info['exec_bin']), *execute_info['exec_argv']]
        except:
            return []


class SimpleExperiment(Experiment):

    def __init__(self, exp_name: str, root=None):
        super().__init__(exp_name, root)
        from . import exphook
        self.set_hook(exphook.LastCmd())
        self.set_hook(exphook.LockFile())
        self.set_hook(exphook.GitCommit())
        self.set_hook(exphook.RecordAbort())
        self.set_hook(exphook.Diary())
        self.set_hook(exphook.TimeMonitor())
        self.set_hook(exphook.FinalReport())
