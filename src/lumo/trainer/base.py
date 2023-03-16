"""
"""
import inspect
import os
import sys
from functools import wraps
from typing import Callable

import numpy as np
import six
import torch
from torch.optim.optimizer import Optimizer

from .. import TrainStage


def _exit_hook_v0(exc_type, exc, tb, *args):
    """Prints the traceback information when an unhandled exception occurs.

    Args:
        exc_type (type): The type of the exception.
        exc (Exception): The instance of the exception.
        tb (traceback): The traceback object containing the call stack.
        *args: Additional arguments to be passed to the function.

    Returns:
        None

    Raises:
        None

    This function is designed to be used as an exit hook with the `sys.excepthook` function.
    It formats the traceback information and removes any lines related to the `_newfunc` function.
    The resulting traceback is printed to the `sys.stderr` stream.

    """
    import traceback
    res = traceback.format_exception(exc_type, exc, tb)
    res = [i for i in res if 'in _newfunc' not in i]
    print(''.join(res), file=sys.stderr)


def _exit_hook(exc_type, exc, tb, *args):
    """Prints an error traceback and displays it using the rich library.

    Args:
        exc_type: Type of the exception that was raised.
        exc: The exception instance that was raised.
        tb: Traceback object that contains information about the exception.
        *args: Optional additional arguments to be passed to _exit_hook_v0.

    Returns:
        None.

    Raises:
        Any exceptions that were not caught by _exit_hook_v0.

    Examples:
        # Call _exit_hook with an exception
        >>> _exit_hook(TypeError, 'test error', traceback, arg1, arg2)

    """
    from rich.console import Console
    console = Console()
    _exit_hook_v0(exc_type, exc, tb, *args)
    try:
        six.reraise(exc_type, exc, tb)
    except:
        from rich.traceback import Traceback
        traceback = Traceback(
            width=100,
            extra_lines=3,
            theme=None,
            word_wrap=False,
            show_locals=False,
            suppress=(),
            max_frames=100,
        )
        for stack in traceback.trace.stacks:
            stack.frames = [i for i in stack.frames if all([i.name != k for k in {'_newfunc'}])]
        console.print(traceback)


init_function = ['icallbacks', 'imodels']
call_dependency = {
    'train': init_function,
}


class _BaseTrainer:
    """Base class for training neural network models.
    """
    __exp_name__ = None

    callback_function = {}

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if cls.__exp_name__ is None:
            cls.__exp_name__ = cls.__name__.lower().replace("trainer", "Exp")

        self._prop = {}
        self._cmp = {}
        self._rev_index = {}
        self._call_order = {}

        self._state_dicts = {
            'optims': set(),
            'models': set(),
            'others': set(),
            'tensor.th': set(),
            'tensor.np': set(),
        }

        from lumo.utils.exithook import replace

        replace(_exit_hook)

        def init_wrapper(func):
            """
            Wraps the train/test/eval functions to initialize in silence.

            Notes:
                Before calling the train/test/eval functions, the `trainer.initialize` method is called,
                 and then the corresponding DataLoader for the stage is initialized through the `process_loader` method.
            """

            @wraps(func)
            def inner(dm=None, params=None, *args, **kwargs):
                """The inner function that wraps the train/test/eval function."""
                init_fn = getattr(self, 'initialize', None)
                if init_fn is not None:
                    init_fn()
                process_loader = getattr(self, 'process_loader', None)
                if process_loader is not None:
                    process_loader(dm, TrainStage.create_from_str(func.__name__))
                func(dm, params, *args, **kwargs)

            return inner

        def cb_wrapper(func, call_set: list):
            """
            Wraps the given function with callback functions.

            Args:
                func (function): The function to wrap.
                call_set (list): A list of callback functions.

            Returns:
                A wrapped function.
            """

            @wraps(func)
            def _newfunc(*aargs, **kkwargs):
                """
                Executes the callback functions before and after the given function and on exception.
                """
                # on_begin
                self._contexts.append(func.__name__)
                for callback in call_set:
                    callback.on_begin(self, func, self.params, *aargs, **kkwargs)
                try:
                    _meter = func(*aargs, **kkwargs)
                except BaseException as e:
                    _handles = [callback.on_exception(self, func, self.params, e, *aargs, **kkwargs)
                                for callback in call_set]
                    self.on_trainer_exception(func, e)
                    if any(_handles):
                        return None
                    else:
                        raise e

                for callback in call_set:
                    callback.on_end(self, func, self.params, _meter, *aargs, **kkwargs)
                self._contexts.pop()
                return _meter

            return _newfunc

        self._contexts = [self.__class__.__name__]
        self.callbacks = []

        vars = dir(self)
        for name in vars:
            if name not in self.callback_function:
                continue
            value = getattr(self, name, None)
            if value is None:
                continue
            if callable(value):
                setattr(self, name, cb_wrapper(value, self.callbacks))

        for func in [self.train, self.test, self.evaluate]:
            setattr(self, func.__name__, init_wrapper(func))
        return self

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key: str, value):
        self.__setattr__(key, value)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name.startswith('_') or name.endswith('_'):
            # when callback is trainer itself, _hooked will be passed, and caused recursion
            # if some pretrained models need to be ignored in save/load stage, it can be named like 'some_' or '_some'
            return

        if isinstance(value, torch.device):
            type_name = 'devices'
        elif isinstance(value, torch.nn.Module):
            type_name = 'models'
        elif isinstance(value, Optimizer):
            type_name = 'optims'
        elif isinstance(value, torch.Tensor):
            type_name = 'tensor.th'
        elif isinstance(value, np.ndarray):
            type_name = 'tensor.np'
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            type_name = 'others'
        else:
            # super().__setattr__(name, value)
            return

        # if name in self.__dict__: TODO workaround multi-gpu error: Expected to mark a variable ready only once
        #     self.__dict__.pop(name)

        self._state_dicts.setdefault(type_name, set()).add(name)
        self._rev_index[name] = type_name

    @property
    def contexts(self):
        """Get the name stack of function call contexts.
        The first is the name of the Trainer class
        """
        return self._contexts

    @property
    def context(self):
        """Get the name of the most recent function call context."""
        return self._contexts[-1]

        # def __getattr__(self, name):

    #     if name.startswith('_') or name.endswith('_'):
    #         # when callback is trainer itself, _hooked will be passed, and caused recursion
    #         # if some pretrained models need to be ignored in save/load stage, it can be named like 'some_' or '_some'
    #         raise AttributeError(name)
    #     type_name = self._rev_index.get(name, None)
    #     if type_name is None:
    #         raise AttributeError(name)
    #
    #     return self._state_dicts[type_name][name]

    @classmethod
    def dirname(cls):
        """Get the directory name of the file where the class is defined.

        Returns:
            A string representing the directory name of the file where the class is defined.
        """
        file = inspect.getfile(cls)
        return os.path.basename(os.path.dirname(file))

    @classmethod
    def filebasename(cls):
        """Get the basename of the file where the class is defined.

        Returns:
            A string representing the basename of the file where the class is defined.
            If an exception occurs, returns 'builtin'.
        """
        try:
            file = inspect.getfile(cls)
            pre = os.path.splitext(os.path.basename(file))[0]
        except:
            pre = 'builtin'
        return pre

    @classmethod
    def generate_exp_name(cls) -> str:
        """Generate an experiment name based on the file basename and the class name.

        Returns:
            A string representing the experiment name, formatted as '<filebasename>.<classname>'.
            If '__exp_name__' is defined, it is used instead of the default class name with 'trainer' replaced by 'exp'.
        """
        pre = cls.filebasename()

        exp_name = cls.__exp_name__
        if exp_name is None:
            exp_name = cls.__name__.lower().replace("trainer", "exp")

        return "{}.{}".format(pre.lower(), exp_name.lower())

    def on_trainer_exception(self, func: Callable, exception: BaseException):
        """Called when an exception occurs during training."""
        pass
