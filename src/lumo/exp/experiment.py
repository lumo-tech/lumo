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
from typing import Any, List, overload
from functools import wraps
from lumo.decorators.process import call_on_main_process_wrap
from lumo.proc import glob
from lumo.proc.dist import is_dist, is_main, local_rank
from lumo.proc.path import blobroot, cache_dir, libhome
from lumo.proc.path import exproot, local_dir
from lumo.utils import safe_io as io
from lumo.utils.fmt import can_be_filename, strftime
from lumo.utils.logger import Logger
from lumo.utils.subprocess import run_command
from .base import BaseExpHook
from ..proc.pid import pid_hash, runtime_pid_obj
from .metric import Metric
from lumo.utils import repository as git_repo


class Experiment:
    """
    Represents an experiment and manages its directory structure. An experiment consists of multiple tests, each of which
    has its own directory to store information related to that test.


    - <cache_root>
        - progress
            - <exp-1>
                - {test-1}.hb
                - {test-1}.pid

    - <exp_root>
        - <exp-1>
            - <test-1> (info_dir)

    - <blob_root>
        - <exp-1>
            - <test-1> (blob_dir)


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

    def __init__(self, exp_name: str = None, test_name=None, roots=None, info_dir=None, blob_dir=None, cache_dir=None):
        """
        Initializes a new instance of the Experiment class.

        Args:
            exp_name (str): The name of the experiment. This should be a legal filename and contain only letters or
                underscores.

        Raises:
            ValueError: If the experiment name is not a legal filename.
        """
        if exp_name is None and info_dir is not None and os.path.exists(info_dir):
            exp = self.__class__.from_disk(info_dir=info_dir)
            self._prop = exp._prop
            self._hooks = exp._hooks
            self._metric = exp._metric
        else:
            assert exp_name is not None
            if not can_be_filename(exp_name):
                raise ValueError(f'Experiment name should be a ligal filename(bettor only contain letter or underline),'
                                 f'but got {exp_name}.')
            self._prop = {'exp_name': exp_name}
            if test_name is None:
                test_name = os.environ.get(Experiment.ENV_TEST_NAME_KEY, None)
            self._prop['test_name'] = test_name
            if roots is None:
                roots = {}
            self._prop['paths'] = roots
            self._prop['note'] = ''

            self._hooks = {}
            self._metric = None

        if info_dir is not None:
            self._prop['info_dir'] = info_dir
        if blob_dir is not None:
            self._prop['blob_dir'] = blob_dir
        if cache_dir is not None:
            self._prop['cache_dir'] = cache_dir
        # wrap
        self.dump_string = self._trigger_change(self.dump_string)
        self.dump_note = self._trigger_change(self.dump_note)
        self.dump_info = self._trigger_change(self.dump_info)
        self.trigger = self._trigger_change(self.trigger)

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
        return f'{self.__class__.__name__}(info_dir="{self.info_dir}")'

    def __str__(self):
        """
        Returns a string representation of the Experiment object.

        Returns:
            str: A string representation of the Experiment object.
        """
        return f'{self.__class__.__name__}(info_dir={self.info_dir})'

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
                flag_fn = os.path.join(self.cache_root, 'dist', f'.{os.getppid()}')
                os.makedirs(os.path.dirname(flag_fn), exist_ok=True)

                if is_main():
                    self._test_name = self._create_test_name(self.exp_dir)

                    with open(flag_fn, 'w') as w:
                        w.write(self._test_name)
                else:
                    time.sleep(random.randint(2, 4))
                    if os.path.exists(flag_fn):
                        with open(flag_fn, 'r') as r:
                            self._test_name = r.readline().strip()
            else:
                self._test_name = self._create_test_name(self.exp_dir)

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
    def properties(self):
        """
        Gets a dictionary containing all properties of the experiment.

        Returns:
            dict: A dictionary containing all properties of the experiment.
        """
        return self._prop

    @property
    def metric(self):
        """
        Gets a dictionary containing all metrics of the experiment.

        Returns:
            Metric: A dictionary containing all metrics of the experiment.
        """
        if self._metric is None:
            self._metric = Metric(self.mk_ipath('metric.pkl'))
            self._metric.dump_metrics = self._trigger_change(self._metric.dump_metrics)
            self._metric.dump_metric = self._trigger_change(self._metric.dump_metric)
        return self._metric

    @property
    def roots(self) -> dict:
        """
        Gets a dictionary containing the paths to various directories associated with the experiment.

        Returns:
            dict: A dictionary containing the paths to various directories associated with the experiment.
        """
        return {
            'info_root': self._prop['paths'].get('info_root', exproot()),
            'cache_root': self._prop['paths'].get('cache_root', cache_dir()),
            'blob_root': self._prop['paths'].get('blob_root', blobroot()),
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

    def trigger(self):
        """trigger the disk change"""
        pass

    def _trigger_change(self, func):
        """
        Decorator function that updates the heartbeat file before executing the decorated function.

        The heartbeat file indicates that a change has occurred in the experiment directory.

        Args:
            func: the function to be decorated.

        Returns:
            A decorated function.
        """

        # test_root update some files
        @wraps(func)
        def inner(*args, **kwargs):
            """wrap function"""
            fn = self.heartbeat_fn
            io.dump_text(self.info_dir, fn)
            io.dump_text(self.info_dir, self.pid_fn)
            return func(*args, **kwargs)

        return inner

    @classmethod
    def _create_test_name(cls, exp_dir):
        """
        Generates a unique test name based on the current date and time.
        regex pattern: [0-9]{6}.[0-9]{3}.[a-z0-9]{3}t

        Returns:
            str: The generated test name.
        """
        from lumo.proc.date import timehash
        from ..utils.fmt import strftime
        fs = os.listdir(exp_dir)
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
        res['last_edit_time'] = strftime()
        if update_from is None:
            res['update_from'] = update_from
        self.dump_info('progress', res, append=True)

    def dump_info(self, key: str, info: Any, append=False):
        """
        Saves information about the experiment to a file.

        Args:
            key (str): The key under which the information will be stored.
            info (Any): The information to store.
            append (bool, optional): Whether to append to the file or overwrite it. Defaults to False.
        """
        fn = self.mk_ipath('info', f'{key}.json')
        if append:
            old_info = self.load_info(key)
            old_info.update(info)
            info = old_info

        self.set_prop(key, info)
        io.dump_json(info, fn)

    def load_info(self, key: str):
        """
        Loads information about the experiment from a file.

        Args:
            key (str): The key under which the information is stored.

        Returns:
            Any: The information stored under the specified key.
        """
        fn = self.mk_ipath('info', f'{key}.json')
        if not os.path.exists(fn):
            return {}
        try:
            return io.load_json(fn)
        except ValueError as e:
            return {}

    def load_note(self):
        """
        Loads the contents of the note file, if it exists.

        Returns:
            A string representing the contents of the note file, or an empty string if the file does not exist.
        """
        fn = self.mk_ipath('note.md')
        if os.path.exists(fn):
            return io.load_text(fn)
        return ''

    def dump_tags(self, *tags):
        """
        Dumps the experiment's tags to the info file.

        Args:
            *tags: a variable-length argument list of tags to be added to the experiment.

        Returns:
            None.
        """
        self.dump_info('tags', tags)

    def dump_note(self, note: str):
        """
        Dumps the contents of the note to the note file.

        Args:
            note: a string representing the contents of the note.

        Returns:
            None.
        """
        fn = self.mk_ipath('note.md')
        self.set_prop('note', note)
        io.dump_text(note, fn)

    def dump_string(self, key: str, info: str, append=False):
        """
        Saves a string to a file.

        Args:
            key (str): The key under which the string will be stored.
            info (str): The string to store.
        """
        fn = self.mk_ipath('text', f'{key}.str')
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
        fn = self.mk_ipath('text', f'{key}.str')
        if not os.path.exists(fn):
            return ''
        return io.load_text(fn)

    def dump_metric(self, key, value, cmp: str, flush=True, **kwargs):
        """
        See Metric for details.
        """
        return self.metric.dump_metric(key, value, cmp, flush, **kwargs)

    def dump_metrics(self, dic: dict, cmp: str):
        """
        See Metric for details.
        """
        return self.metric.dump_metrics(dic, cmp)

    @property
    def info_root(self):
        return self.roots['info_root']

    @property
    def cache_root(self):
        return self.roots['cache_root']

    @property
    def blob_root(self):
        return self.roots['blob_root']

    @property
    def pid_fn(self):
        fn = os.path.join(self.cache_root, 'pid', self.exp_name, f'{self.test_name}.pid')
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        return fn

    @property
    def heartbeat_fn(self):
        fn = os.path.join(self.cache_root, 'heartbeat', self.exp_name, f'{self.test_name}.hb')
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        return fn

    @property
    def exp_dir(self):
        d = os.path.join(self.info_root, self.exp_name)
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def info_dir(self):
        d = self.properties.get('info_dir')
        if d is None:
            d = os.path.join(self.info_root, self.exp_name, self.test_name)
            os.makedirs(d, exist_ok=True)
        return d

    @property
    def cache_dir(self):
        d = self.properties.get('cache_dir')
        if d is None:
            d = os.path.join(self.cache_root, self.exp_name, self.test_name)
            os.makedirs(d, exist_ok=True)
        return d

    @property
    def blob_dir(self):
        d = self.properties.get('blob_dir')
        if d is None:
            d = os.path.join(self.blob_root, self.exp_name, self.test_name)
            os.makedirs(d, exist_ok=True)
        return d

    def _mk_path(self, *path: str, is_dir: bool) -> str:
        """
        Helper method to create a directory path if it does not exist and return the path.

        Args:
            *path: tuple of path strings to be joined.
            is_dir: boolean flag indicating whether the path is a directory.

        Returns:
            str: the full path created.
        """
        path = os.path.join(*path)
        if is_dir:
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def mk_ipath(self, *path: str, is_dir: bool = False) -> str:
        """
        Creates a directory path within the experiment's info directory.

        Args:
            *path: tuple of path strings to be joined.
            is_dir: boolean flag indicating whether the path is a directory. Default is False.

        Returns:
            str: the full path created.
        """
        return self._mk_path(self.info_dir, *path, is_dir=is_dir)

    def mk_cpath(self, *path: str, is_dir: bool = False) -> str:
        """
        Creates a directory path within the experiment's cache directory.

        Args:
            *path: tuple of path strings to be joined.
            is_dir: boolean flag indicating whether the path is a directory. Default is False.

        Returns:
            str: the full path created.
        """
        return self._mk_path(self.cache_dir, *path, is_dir=is_dir)

    def mk_bpath(self, *path: str, is_dir: bool = False) -> str:
        """
        Creates a directory path within the experiment's blob directory.

        Args:
            *path: tuple of path strings to be joined.
            is_dir: boolean flag indicating whether the path is a directory. Default is False.

        Returns:
            str: the full path created.
        """
        return self._mk_path(self.blob_dir, *path, is_dir=is_dir)

    def mk_rpath(self, *path: str, is_dir: bool = False) -> str:
        """
        Creates a directory path within the user's home directory.

        Args:
            *path: tuple of path strings to be joined.
            is_dir: boolean flag indicating whether the path is a directory. Default is False.

        Returns:
            str: the full path created.
        """
        return self._mk_path(libhome(), *path, is_dir=is_dir)

    @classmethod
    @property
    def Class(cls):
        return cls

    def rerun(self, arg_list: List[str]):
        """rerun this test in another"""
        # self.properties['']
        new_test_name = self._create_test_name(self.exp_dir)
        new_exp = Experiment(self.exp_name, test_name=new_test_name)
        self.dump_info('rerun', {'rerun_at': {new_exp.test_name: True}}, append=True)
        old_rerun_info = self.properties.get('rerun', {})
        count = 1
        if isinstance(old_rerun_info, dict):
            count += old_rerun_info.get('repeat', 0)
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
        self.dump_info('exp_name', self.exp_name)
        self.dump_info('test_name', self.test_name)
        self.dump_info('roots', self.roots)

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

        self.dump_tags()

        # register start
        self.dump_info('progress', {'start': strftime()}, append=True)
        self.dump_progress(0)
        # register progress
        self.proc = run_command(f'python3 -m lumo.exp.agent --info_dir={self.info_dir}', non_block=True)

        self.add_exit_hook(self.end)

    @call_on_main_process_wrap
    def start(self):
        """
        Starts the experiment.
        """
        if self.properties.get('progress', None) is not None:
            return
        self.initial()
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
        if self.properties.get('progress', None) is None:
            return
        if self.properties['progress'].get('end', False):
            return

        if end_code == 0:
            self.dump_progress(1)

        self.dump_info('progress', {'end': strftime(), 'end_code': end_code}, append=True)
        for hook in self._hooks.values():  # type: BaseExpHook
            hook.on_end(self, end_code=end_code, *args, **extra)

        self.proc.terminate()
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
            func()

        atexit.register(exp_func)

    @classmethod
    def from_cache(cls, dic: dict):
        """
        Create an Experiment object from cached Experiment data.

        If the disk has been modified (as detected by `~Experiment.heartbeat_fn`),
        reload the Experiment from disk. Otherwise, create a new Experiment
        object with the cached data.

        Args:
            cls: The Experiment class.
            dic: A dictionary containing cached Experiment data.

        Returns:
            An Experiment object.
        """
        paths = dic.pop('paths', {})
        _ = dic.pop('metrics')
        self = cls(exp_name=dic['exp_name'], test_name=dic['test_name'], roots=paths)
        if os.path.exists(self.heartbeat_fn):
            return cls.from_disk(self.info_dir)
        self._prop.update(dic)

        return self

    @classmethod
    def from_disk(cls, info_dir, blob_dir=None, cache_dir=None):
        """
        Creates an Experiment object from a test root directory on disk.

        Args:
            info_dir (str): The path to the test root directory.

        Returns:
            Experiment: An Experiment object created from the test root directory.

        Raises:
            ValueError: If the path is not a valid test root directory.
        """
        from lumo.exp.watch import is_test_root
        if not is_test_root(info_dir):
            raise ValueError(f'{info_dir} is not a valid test_root')
        info_dir = os.path.abspath(info_dir)
        exp_name = io.load_json(os.path.join(info_dir, 'info', 'exp_name.json'))
        test_name = io.load_json(os.path.join(info_dir, 'info', 'test_name.json'))

        paths_fn = os.path.join(info_dir, 'info', f'paths.json')
        if os.path.exists(paths_fn):
            try:
                paths = io.load_json(paths_fn)
            except ValueError as e:
                paths = {}
        else:
            paths = {}

        # given exp_name will stop __init__ load information by .from_disk()
        self = cls(exp_name, test_name=test_name, roots=paths,
                   info_dir=info_dir, blob_dir=blob_dir, cache_dir=cache_dir)

        # load prop
        for f in os.listdir(self.mk_ipath('info', is_dir=True)):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_info(key))

        for f in os.listdir(self.mk_ipath('text', is_dir=True)):
            key = os.path.splitext(f)[0]
            self.set_prop(key, self.load_string(key))

        self.set_prop('note', self.load_note())

        return self

    def cache(self):
        """Cache information of current test."""
        return {
            **self.properties,
            'metrics': self.metric.value,
        }

    def dict(self):
        """Get full information of current test, including dynamic status."""
        return {
            **self.properties,
            'is_alive': self.is_alive,
            'metrics': self.metric.value,
        }

    @overload
    def backup(self, backend: str = 'local',
               target_dir: str = None, with_code=False, with_blob=False, with_cache=False):
        ...

    @overload
    def backup(self, backend: str = 'github', access_token: str = None,
               labels: list = None, update: bool = True,
               **kwargs):
        ...

    def backup(self, backend: str = 'github', **kwargs):
        """
        Backup this experiment into the given target, currently only support GitHub, you can implement your own way
        by the provided information of Experiment.
        """
        from .backup import backup_regist
        return backup_regist[backend](exp=self, **kwargs)

    def archive(self, target_dir=None):
        if 'git' not in self.properties:
            return None

        if target_dir is None:
            target_dir = self.blob_dir

        repo = git_repo.load_repo(self.project_root)
        return git_repo.git_archive(target_dir, repo, self.properties['git']['commit'])


class SimpleExperiment(Experiment):
    """
    A simple to use experiment subclass that extends the base `Experiment` class and sets up some useful hooks to
    execute before and after the experiment.
    """

    def __init__(self, exp_name: str = None, test_name=None, roots=None, info_dir=None):
        super().__init__(exp_name, test_name, roots, info_dir)
        from . import exphook
        self.set_hook(exphook.LastCmd())
        self.set_hook(exphook.LockFile())
        self.set_hook(exphook.GitCommit())
        self.set_hook(exphook.RecordAbort())
        self.set_hook(exphook.Diary())
        # self.set_hook(exphook.TimeMonitor())
        self.set_hook(exphook.FinalReport())
