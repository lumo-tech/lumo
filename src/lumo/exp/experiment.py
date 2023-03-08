import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Union, Any

from lumo.decorators.process import call_on_main_process_wrap
from lumo.proc import glob
from lumo.proc.dist import is_dist, is_main, local_rank
from lumo.proc.path import blobroot, libhome, progressroot
from lumo.proc.path import exproot, local_dir
from lumo.utils import safe_io as io
from lumo.utils.fmt import can_be_filename
from lumo.utils.logger import Logger
from .base import BaseExpHook
from ..proc.pid import pid_hash, runtime_pid_obj


def checkdir(path: Union[Path, str]):
    """
    Create a directory at the specified path if it does not already exist.

    Args:
        path (Union[Path, str]): The path to the directory to be created.

    Returns:
        Path: The Path object representing the created directory.
    """
    if isinstance(path, str):
        os.makedirs(path, exist_ok=True)
    elif isinstance(path, Path):
        path.mkdir(parents=True, exist_ok=True)
    return path


class Experiment:
    """
    Represents an experiment and manages its directory structure. An experiment consists of multiple tests, each of which
    has its own directory to store information related to that test.

    (By default), the directory structure is as following:
    .lumo (libroot)
        - progress
            - ".{pid}" -> hash
                if pid exists and hash(psutil.Process) == hash in file: is run
                else: is closed
        - experiments # (exp_root) record information (e.g., .log, params files, etc.)
            - {experiment-name-1}
                - {test-1}
                    # infomation
                    {
                        progress
                        pid_hash (for lumo.client monitor)
                        other_info: git, file, version_lock, etc.
                    }
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
        """
        Initializes a new instance of the Experiment class.

        Args:
            exp_name (str): The name of the experiment. This should be a legal filename and contain only letters or
                underscores.
            root (str, optional): The root directory where the experiment's directories will be created. Defaults to
                None, in which case the root directory is set to the library's home directory.

        Raises:
            ValueError: If the experiment name is not a legal filename.
        """
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
        """
        str: Gets the name of the experiment.
        """
        return self._prop['exp_name']

    @property
    def _test_name(self):
        """
        str: Gets the name of the current test being run.
        """
        return self._prop.get('test_name', None)

    @_test_name.setter
    def _test_name(self, value):
        """
         Sets the name of the current test being run.

         Args:
             value (str): The name of the current test.
        """
        self._prop['test_name'] = value

    @property
    def test_name_with_dist(self):
        """
        str: Gets the name of the current test with the local rank number
        appended to it if running in distributed mode.

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
        """
        str: Gets the name of the current test being run.

        If the test name is not set, generates a new unique name and sets it.
        """
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
        """
        Generates a unique test name based on the current date and time.
        regex pattern: [0-9]{6}.[0-9]{3}.[a-z0-9]{3}t

        Returns:
            str: The generated test name.
        """
        from lumo.proc.date import timehash
        from ..utils.fmt import strftime
        fs = os.listdir(self.exp_root)
        date_str = strftime('%y%m%d')
        fs = [i for i in fs if i.startswith(date_str)]
        _test_name = f"{date_str}.{len(fs):03d}.{timehash()[-6:-4]}t"
        return _test_name

    @property
    def root_branch(self):
        """
        Path: Gets the root branch directory of the experiment.
        """
        val = self._root
        return checkdir(val)

    @property
    def lib_root(self):
        """
        str: Gets the path of the library's root directory.
        """
        return self.root_branch.as_posix()

    @property
    def exp_branch(self):
        """
        Path: Gets the experiment branch directory.
        """
        val = Path(exproot()).joinpath(self.exp_name)
        return checkdir(val)

    @property
    def blob_branch(self):
        """
        Path: Gets the blob branch directory, which is used to store big binary files like model state dicts.
        """
        val = Path(blobroot()).joinpath(self.exp_name, self.test_name)
        return checkdir(val)

    @property
    def progress_branch(self):
        """
        Path: Gets the progress branch directory, which is used to store progress information about running processes.
        """
        val = Path(progressroot())
        return checkdir(val)

    @property
    def test_branch(self):
        """
        Path: Gets the test branch directory, which is used to store information related to the current test being run.
        """
        val = self.exp_branch.joinpath(self.test_name)
        return checkdir(val)

    def dump_progress(self, ratio: float, update_from=None):
        """
        Saves progress information about the experiment.

        Args:
            ratio (float): The progress ratio as a number between 0 and 1.
            update_from: The process from which the progress update came from.
        """
        res = {'ratio': max(min(ratio, 1), 0)}
        if update_from is None:
            res['update_from'] = update_from
        self.dump_info('progress', res, append=True)

    def dump_info(self, key: str, info: Any, append=False, info_dir='info', set_prop=True):
        """
        Saves information about the experiment to a file.

        Args:
            key (str): The key under which the information will be stored.
            info (Any): The information to store.
            append (bool, optional): Whether to append to the file or overwrite it. Defaults to False.
            info_dir (str, optional): The name of the directory where the file will be stored. Defaults to 'info'.
            set_prop (bool, optional): Whether to set the experiment property with the same key to the saved information.
                Defaults to True.
        """
        fn = self.test_file(f'{key}.json', info_dir)
        if append:
            old_info = self.load_info(key, info_dir=info_dir)
            old_info.update(info)
            info = old_info
        if set_prop:
            self.set_prop(key, info)
        io.dump_json(info, fn)

    def load_info(self, key: str, info_dir='info'):
        """
        Loads information about the experiment from a file.

        Args:
            key (str): The key under which the information is stored.
            info_dir (str, optional): The name of the directory where the file is stored. Defaults to 'info'.

        Returns:
            Any: The information stored under the specified key.
        """
        fn = self.test_file(f'{key}.json', info_dir)
        if not os.path.exists(fn):
            return {}
        return io.load_json(fn)

    def dump_string(self, key: str, info: str):
        """
        Saves a string to a file.

        Args:
            key (str): The key under which the string will be stored.
            info (str): The string to store.
        """
        fn = self.test_file(f'{key}.str', 'text')
        io.dump_text(info, fn)
        self.set_prop(key, info)

    def load_string(self, key: str):
        """
        Loads a string from a file.

        Args:
            key (str): The key under which the string is stored.

        Returns:
            str: The string stored under the specified key.
        """
        fn = self.test_file(f'{key}.str', 'text')
        if not os.path.exists(fn):
            return ''
        return io.load_text(fn)

    @property
    def tags(self):
        """
        dict: Gets the tags associated with the experiment.
        """
        tags = {}
        for path in self.test_branch.joinpath('tags').glob('tag.*.json'):
            ptags = io.load_json(path.as_posix())  # type: dict
            tags.setdefault(path.suffixes[0].strip('.'), []).extend(ptags.keys())
        return tags

    def add_tag(self, tag: str, name_space: str = 'default'):
        """
        Adds a tag to the experiment.
        Args:
            tag (str): The tag to add.
            name_space (str, optional): The namespace under which to
                add the tag. Defaults to 'default'.
        """
        self.dump_info(f'tag.{name_space}', {
            tag: None
        }, append=True, info_dir='tags', set_prop=False)

    def exp_file(self, filename, *args):
        """
        Gets the path to a file in the experiment directory.

        Args:
            filename (str): The name of the file.
            *args: Any additional subdirectory names to include in the file path.

        Returns:
            str: The path to the specified file.
        """
        parent = self.exp_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def test_file(self, filename, *args):
        """
        Gets the path to a file in the test directory.

        Args:
            filename (str): The name of the file.
            *args: Any additional subdirectory names to include in the file path.

        Returns:
            str: The path to the specified file.
        """
        parent = self.test_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def exp_dir(self, *args):
        """
        Gets the path to a directory in the experiment directory.

        Args:
            *args: Any subdirectory names to include in the directory path.

        Returns:
            str: The path to the specified directory.
        """
        parent = self.exp_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def root_file(self, filename, *args):
        """
        Gets the path to a file in the library's root directory.

        Args:
            filename (str): The name of the file.
            *args: Any additional subdirectory names to include in the file path.

        Returns:
            str: The path to the specified file.
        """
        parent = self.root_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def root_dir(self, *args):
        """
        Gets the path to a directory in the library's root directory.

        Args:
            *args: Any subdirectory names to include in the directory path.

        Returns:
            str: The path to the specified directory.
        """
        parent = self.root_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def test_dir(self, *args):
        """
        Gets the path to a directory in the test directory.

        Args:
            *args: Any subdirectory names to include in the directory path.

        Returns:
            str: The path to the specified directory.
        """
        parent = self.test_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def blob_file(self, filename, *args):
        """
        Gets the path to a file in the blob directory.

        Args:
            filename (str): The name of the file.
            *args: Any additional subdirectory names to include in the file path.

        Returns:
            str: The path to the specified file.
        """
        parent = self.blob_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    def progress_file(self, filename):
        """
        Gets the path to a file in the progress directory.

        Args:
            filename (str): The name of the file.

        Returns:
            str: The path to the specified file.
        """
        return self.progress_branch.joinpath(filename).as_posix()

    def blob_dir(self, *args):
        """
        Gets the path to a directory in the blob directory.
        Args:
            *args: Any subdirectory names to include in the directory path.

        Returns:
            str: The path to the specified directory.
        """
        parent = self.blob_branch.joinpath(*args)
        return checkdir(parent).as_posix()

    def __enter__(self):
        """
        Starts the experiment when the Experiment object is used as a context manager using the 'with' statement.

        Returns:
            Experiment: The Experiment object.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ends the experiment when the 'with' statement block exits.

        Args:
            exc_type (type): The type of the exception that occurred, if any.
            exc_val (Exception): The exception object that was raised, if any.
            exc_tb (traceback): The traceback object for the exception, if any.
        """
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
        """
        Registers a function to be called when the program exits.

        Args:
            func (callable): The function to register.
        """
        import atexit
        def exp_func():
            """Function executed before process exit."""
            func(self)

        atexit.register(exp_func)

    @call_on_main_process_wrap
    def initial(self):
        """
        Initializes the experiment by setting up progress, information, and PID tracking.
        """
        self.add_tag(self.__class__.__name__, 'exp_type')
        self.dump_progress(0)
        self.dump_info('execute', {
            'repo': self.project_root,
            'cwd': os.getcwd(),
            'exec_file': sys.argv[0],
            'exec_bin': sys.executable,
            'exec_argv': sys.argv
        })
        self.dump_info('pinfo', {
            'pid': os.getpid(),
            'hash': pid_hash(),
            'obj': runtime_pid_obj(),
        })

        # register progress
        io.dump_text(self.test_root, self.progress_file(f'{os.getpid()}'))

    @call_on_main_process_wrap
    def start(self):
        """
        Starts the experiment.
        """
        if self.get_prop('start', False):
            return
        self.initial()
        self.dump_info('start', True)
        for hook in self._hooks.values():  # type: BaseExpHook
            hook.on_start(self)
        return self

    @call_on_main_process_wrap
    def end(self, end_code=0, *args, **extra):
        """
        Ends the experiment.

        Args:
            end_code (int): The exit code to set for the experiment.
            *args: Additional arguments to pass to the end hooks.
            **extra: Additional keyword arguments to pass to the end hooks.
        """
        if not self.get_prop('start', False):
            return
        if self.get_prop('end', False):
            return
        self.dump_progress(1)
        self.dump_info('end', True)
        for hook in self._hooks.values():  # type: BaseExpHook
            hook.on_end(self, end_code=end_code, *args, **extra)
        return self

    @property
    def repo_name(self):
        """
        Gets the name of the repository associated with the experiment.

        Returns:
            str: The name of the repository.
        """
        return self.project_name

    @property
    def project_name(self):
        """
        Gets the name of the project associated with the experiment.

        Returns:
            str: The name of the project.
        """
        return os.path.basename(self.project_root)

    @property
    def project_root(self):
        """
        Gets the path to the root directory of the project associated with the experiment.

        Returns:
            str: The path to the root directory of the project.
        """
        return local_dir()

    @property
    def exp_root(self):
        """
        Gets the path to the directory containing the tests for the experiment.

        Returns:
            str: The path to the experiment root directory.
        """
        return self.exp_branch.as_posix()

    @property
    def test_root(self):
        """
        Gets the path to the directory containing information about the current test.

        Returns:
            str: The path to the test root directory.
        """
        return self.test_branch.as_posix()

    @property
    def blob_root(self):
        """
        Gets the path to the directory containing large binary files associated with the experiment.

        Returns:
            str: The path to the blob root directory.
        """
        return self.blob_branch.as_posix()

    def __getitem__(self, item):
        """
        Gets a property of the experiment.

        Args:
            item (str): The name of the property to get.

        Returns:
            Any: The value of the property.
        """
        return self._prop[item]

    def __setitem__(self, key, value):
        """
        Sets a property of the experiment.

        Args:
            key (str): The name of the property to set.
            value (Any): The value to set the property to.
        """
        self._prop[key] = value

    def get_prop(self, key, default=None):
        """
        Gets the value of a property of the experiment.

        Args:
            key (str): The name of the property to get.
            default (Any, optional): The default value to return if the property does not exist. Defaults to None.

        Returns:
            Any: The value of the property, or the default value if the property does not exist.
        """
        return self._prop.get(key, default)

    def has_prop(self, key):
        """
        Determines whether the experiment has a certain property.

        Args:
            key (str): The name of the property to check for.

        Returns:
            bool: True if the experiment has the property, False otherwise.
        """
        return key in self._prop

    def set_prop(self, key, value):
        """
        Sets a property of the experiment.

        Args:
            key (str): The name of the property to set.
            value (Any): The value to set the property to.
        """
        self._prop[key] = value

    @property
    def properties(self):
        """
        Gets a dictionary containing all properties of the experiment.

        Returns:
            dict: A dictionary containing all properties of the experiment.
        """
        return self._prop

    @property
    def paths(self) -> dict:
        """
        Gets a dictionary containing the paths to various directories associated with the experiment.

        Returns:
            dict: A dictionary containing the paths to various directories associated with the experiment.
        """
        return {
            'root': self.root_branch.as_posix(),
            'exp_root': self.exp_root,
            'test_root': self.test_root,
            'blob_root': self.blob_root,
        }

    @property
    def enable_properties(self) -> set:
        """
        Gets a set of the names of all properties that have been set for the experiment.

        Returns:
            set: A set of the names of all properties that have been set for the experiment.
        """
        return set(self._prop.keys())

    @call_on_main_process_wrap
    def set_hook(self, hook: BaseExpHook):
        """
        Registers a hook to be executed during the experiment.

        Args:
            hook (BaseExpHook): The hook to register.
        """
        hook.regist(self)
        if not glob.get(hook.config_name, True):
            self.dump_info(hook.name, {
                'code': -1,
                'msg': f'{hook.name} disabled'
            })
            return self
        self.logger.info(f'Register {hook}.')
        self._hooks[hook.__class__.__name__] = hook
        self.add_tag(hook.__class__.__name__, 'hooks')
        return self

    def load_prop(self):
        """
        Loads all properties associated with the experiment from disk.
        """
        for f in os.listdir(self.test_dir('info')):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_info(key))

        for f in os.listdir(self.test_dir('text')):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_string(key))

    @classmethod
    def from_disk(cls, path):
        """
        Creates an Experiment object from a test root directory on disk.

        Args:
            path (str): The path to the test root directory.

        Returns:
            Experiment: An Experiment object created from the test root directory.

        Raises:
            ValueError: If the path is not a valid test root directory.
        """
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
    def is_alive(self):
        """
        Determines whether the process associated with the experiment is still running.

        Returns:
            bool: True if the process is still running, False otherwise.
        """
        pinfo = self.properties['pinfo']

        hash_obj = runtime_pid_obj(pinfo['pid'])
        if hash_obj is None:
            return False

        return pid_hash(hash_obj) == pinfo['hash']

    @property
    def exec_argv(self):
        """
        Gets the arguments used to execute the script associated with the experiment.

        Returns:
            List[str]: A list of arguments used to execute the script.
        """
        execute_info = self.get_prop('execute')
        try:
            return [os.path.basename(execute_info['exec_bin']), *execute_info['exec_argv']]
        except:
            return []

    def __repr__(self):
        """
        Returns a string representation of the Experiment object.

        Returns:
            str: A string representation of the Experiment object.
        """
        return f'{self.exp_name}->({self.test_name})'

    def __str__(self):
        """
        Returns a string representation of the Experiment object.

        Returns:
            str: A string representation of the Experiment object.
        """
        return self.__repr__()


class SimpleExperiment(Experiment):
    """
    A simple to use experiment subclass that extends the base `Experiment` class and sets up some useful hooks to
    execute before and after the experiment.
    """

    def __init__(self, exp_name: str, root=None):
        super().__init__(exp_name, root)
        from . import exphook
        self.set_hook(exphook.LastCmd())
        self.set_hook(exphook.LockFile())
        self.set_hook(exphook.GitCommit())
        self.set_hook(exphook.RecordAbort())
        self.set_hook(exphook.Diary())
        # self.set_hook(exphook.TimeMonitor())
        self.set_hook(exphook.FinalReport())
