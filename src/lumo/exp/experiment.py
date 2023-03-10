"""
Experiment 负责的内容
 - 管理路径 PathHelper
 - 记录信息 InfoIO 和度量 Metric
 - 快照 snap 和复现 rerun
"""
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Union, Any, List
from functools import wraps
from lumo.decorators.process import call_on_main_process_wrap
from lumo.proc import glob
from lumo.proc.dist import is_dist, is_main, local_rank
from lumo.proc.path import blobroot, libhome, progressroot
from lumo.proc.path import exproot, local_dir
from lumo.utils import safe_io as io
from lumo.utils.fmt import can_be_filename, strftime
from lumo.utils.logger import Logger
from .base import BaseExpHook
from ..proc.pid import pid_hash, runtime_pid_obj
from .metric import Metric


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
                    metric_board.sqlite (metrics in training ,powered by dbrecord)
                    metric.pkl (final metrics)
                    params.yaml (hyper parameter)
                    note.md (manually note)
                    l.0.2303062216.log (log file)
                    text/
                        exception.str
                        ...
                    info/
                        *.json
                        git.json
                        execute.json
                        lock.json
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
    {lumo.cache_dir}
        - progress # (metric_root) record metrics (by trainer.database)
            - hb  # trigger for test information update
                {experiment-name}
                - {test-1} -> timestamp
                - {test-2} -> timestamp
            - pid # link to running process
                - {pid1} -> test_root
                - {pid2} -> test_root
    """

    ENV_TEST_NAME_KEY = 'LUMO_EXP_TEST_NAME'

    def __init__(self, exp_name: str, root=None, test_name=None):
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
        if test_name is None:
            test_name = os.environ.get(Experiment.ENV_TEST_NAME_KEY, None)
        self._prop['test_name'] = test_name
        self._hooks = {}

        self._metric = Metric(self.metrics_fn)

        # wrap
        self._metric.dump_metrics = self._trigger_change(self._metric.dump_metrics)
        self._metric.dump_metric = self._trigger_change(self._metric.dump_metric)
        self.dump_string = self._trigger_change(self.dump_string)
        self.dump_note = self._trigger_change(self.dump_note)
        self.dump_info = self._trigger_change(self.dump_info)

        if root is None:
            root = libhome()
        self._root = Path(os.path.abspath(root))
        self.add_exit_hook(self.end)
        self.logger = Logger()

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

    def _repr_html_(self):
        """Return a html representation for a particular DataFrame."""
        return self.__repr__()

    @property
    def exp_name(self):
        """
        str: Gets the name of the experiment.
        """
        return self._prop['exp_name']

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
                    self._test_name = self._create_test_name(self.exp_root)
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
                self._test_name = self._create_test_name(self.exp_root)

        return self._test_name

    @property
    def _test_name(self):
        """
        str: Gets the name of the current test being run.
        """
        return self._prop.get('test_name')

    @_test_name.setter
    def _test_name(self, value):
        """
         Sets the name of the current test being run.

         Args:
             value (str): The name of the current test.
        """
        self._prop['test_name'] = value

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

    @property
    def properties(self):
        """
        Gets a dictionary containing all properties of the experiment.

        Returns:
            dict: A dictionary containing all properties of the experiment.
        """
        return self._prop

    @property
    def metrics_fn(self):
        return self.test_file('metric.pkl')

    @property
    def metric(self):
        """
        Gets a dictionary containing all metrics of the experiment.

        Returns:
            Metric: A dictionary containing all metrics of the experiment.
        """
        return self._metric

    @property
    def note_fn(self):
        fn = self.test_file('note.md')
        if os.path.exists(fn):
            return io.load_text(fn)
        return fn

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
        execute_info = self.properties.get('execute')
        try:
            return [os.path.basename(execute_info['exec_bin']), *execute_info['exec_argv']]
        except:
            return []

    def _trigger_change(self, func):
        #  test_root update some files
        @wraps(func)
        def inner(*args, **kwargs):
            fn = self.progress_file(f'{self.test_name}.heartbeat', 'hb', self.exp_name)
            io.dump_text(self.test_root, fn)
            func(*args, **kwargs)

        return inner

    @classmethod
    def _create_test_name(cls, exp_root):
        """
        Generates a unique test name based on the current date and time.
        regex pattern: [0-9]{6}.[0-9]{3}.[a-z0-9]{3}t

        Returns:
            str: The generated test name.
        """
        from lumo.proc.date import timehash
        from ..utils.fmt import strftime
        fs = os.listdir(exp_root)
        date_str = strftime('%y%m%d')
        fs = [i for i in fs if i.startswith(date_str)]
        _test_name = f"{date_str}.{len(fs):03d}.{timehash()[-6:-4]}t"
        return _test_name

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
            res['last_edit_time'] = strftime()
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
            self[key] = info
            # self.set_prop(key, info)
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
        try:
            return io.load_json(fn)
        except ValueError as e:
            return {}

    def load_note(self):
        fn = self.test_file('note.md')
        if os.path.exists(fn):
            return io.load_text(fn)
        return ''

    def dump_tags(self, *tags):
        self.dump_info('tags', tags)

    def dump_note(self, note: str):
        fn = self.test_file('note.md')
        self.set_prop('note', note)
        io.dump_text(note, fn)

    def dump_string(self, key: str, info: str, append=False):
        """
        Saves a string to a file.

        Args:
            key (str): The key under which the string will be stored.
            info (str): The string to store.
        """
        fn = self.test_file(f'{key}.str', 'text')
        io.dump_text(info, fn, append=append)
        if not append:
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

    def dump_metric(self, key, value, cmp: str, flush=True, **kwargs):
        return self.metric.dump_metric(key, value, cmp, flush, **kwargs)

    def dump_metrics(self, dic: dict, cmp: str):
        return self.metric.dump_metrics(dic, cmp)

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

    def progress_file(self, filename, *args):
        """
        Gets the path to a file in the progress directory.

        Args:
            filename (str): The name of the file.

        Returns:
            str: The path to the specified file.
        """
        parent = self.progress_branch.joinpath(*args)
        return checkdir(parent).joinpath(filename).as_posix()

    @classmethod
    @property
    def Class(cls):
        return cls

    def rerun(self, arg_list: List[str]):
        """rerun this test in another"""
        # self.properties['']
        new_test_name = self._create_test_name(self.exp_root)
        new_exp = Experiment(self.exp_name, root=self._root, test_name=new_test_name)
        self.dump_info('deprecated', {'rerun_at': {new_exp.test_name: True}}, append=True)
        old_rerun_info = self.properties.get('rerun', None)
        count = 1
        if old_rerun_info is not None:
            count += old_rerun_info['count']
        new_exp.dump_info('rerun', {'from': self.test_name, 'repeat': count})
        from lumo.utils.subprocess import run_command
        old_exec = self.properties['execute']
        command = ' '.join([old_exec['exec_bin'], old_exec['exec_file'], *old_exec['exec_argv'], *arg_list])
        env = os.environ.copy()
        env[Experiment.ENV_TEST_NAME_KEY] = new_exp.test_name
        return run_command(command, cwd=old_exec['cwd'], env=env)

    @call_on_main_process_wrap
    def initial(self):
        """
        Initializes the experiment by setting up progress, information, and PID tracking.
        """
        self.dump_info('progress', {'start': strftime(), 'finished': False}, append=True)
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
        # register this process
        io.dump_text(self.test_root, self.progress_file(f'{os.getpid()}', 'pid'))

    @call_on_main_process_wrap
    def start(self):
        """
        Starts the experiment.
        """
        if self.properties.get('start', False):
            return
        self.initial()
        self.set_prop('start', True)
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
        if not self.is_alive:
            return
        if not self.properties.get('start', False):
            return
        if self.properties.get('end', False):
            return
        self.set_prop('end', True)
        self.dump_progress(1)

        self.dump_info('progress', {'end': strftime(), 'finished': end_code == 0}, append=True)
        for hook in self._hooks.values():  # type: BaseExpHook
            hook.on_end(self, end_code=end_code, *args, **extra)
        return self

    @call_on_main_process_wrap
    def set_hook(self, hook: BaseExpHook):
        """
        Registers a hook to be executed during the experiment.

        Args:
            hook (BaseExpHook): The hook to register.
        """
        if not glob.get(hook.config_name, True):
            self.dump_info('hooks', {
                hook.__class__.__name__: {'loaded': False, 'msg': 'disabled by config'}
            }, append=True)
            return self
        else:
            hook.regist(self)
            self.dump_info('hooks', {
                hook.__class__.__name__: {'loaded': True, 'msg': ''}
            }, append=True)
            self.logger.info(f'Register {hook}.')
            self._hooks[hook.__class__.__name__] = hook
            return self

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
        self = cls(test_root.parent.name, root=root, test_name=test_root.name)

        # load prop
        for f in os.listdir(self.test_dir('info')):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_info(key))

        for f in os.listdir(self.test_dir('text')):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_string(key))

        self.set_prop('note', self.load_note())

        # load metric
        self._metric = Metric(self.metrics_fn)
        return self

    def dict(self):
        return {
            'path': {
                'test_root': self.test_root,
                'exp_root': self.exp_root,
                'blob_root': self.blob_root,
            },
            **self.properties,
            'is_alive': self.is_alive,
            'metrics': self.metric.value,
        }


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
