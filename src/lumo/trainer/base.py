"""
 - 提供主要的训练流
    - 加载数据集 DataLoader
    - train、test、eval
 - 提供对训练流的控制、回调
    - callbacks
 - 提供对训练中间的状态保存
    - metric: Meter
    - checkpoints: Saver


trainer = Trainer()
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

from .components import TrainerPropVar
from .. import TrainStage


def _exit_hook_v0(exc_type, exc, tb, *args):
    import traceback
    res = traceback.format_exception(exc_type, exc, tb)
    res = [i for i in res if 'in _newfunc' not in i]
    print(''.join(res), file=sys.stderr)


def _exit_hook(exc_type, exc, tb, *args):
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


def wrapper(self, func, _call_set: list):
    """
    对每个 Trainer 的 _call_backs 类变量中定义的函数尝试绑定回调
    Args:
        func:
        _call_set:

    Returns:

    """

    @wraps(func)
    def _newfunc(*aargs, **kkwargs):
        """执行前回调 on_begin() 、执行后回调 on_end()、执行异常则回调 on_exception() """
        for callback in _call_set:
            callback.on_begin(self, func, self.params, *aargs, **kkwargs)
        try:
            _meter = func(*aargs, **kkwargs)
        except BaseException as e:
            _handles = [callback.on_exception(self, func, self.params, e, *aargs, **kkwargs)
                        for callback in _call_set]

            if any(_handles):
                return None
            else:
                raise e

        for callback in _call_set:
            callback.on_end(self, func, self.params, _meter, *aargs, **kkwargs)
        return _meter

    return _newfunc


init_function = ['icallbacks', 'imodels']
call_dependency = {
    'train': init_function,
}


class _BaseTrainer:
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
            @wraps(func)
            def inner(dm=None, params=None, *args, **kwargs):
                init_fn = getattr(self, 'initialize', None)
                if init_fn is not None:
                    init_fn()
                process_loader = getattr(self, 'process_loader', None)
                if process_loader is not None:
                    process_loader(dm, TrainStage.create_from_str(func.__name__))
                func(*args, **kwargs)

            return inner

        def cb_wrapper(func, call_set: list):
            """
            对每个 Trainer 的 _call_backs 类变量中定义的函数尝试绑定回调
            Args:
                func:
                call_set:

            Returns:

            """

            @wraps(func)
            def _newfunc(*aargs, **kkwargs):
                """执行前回调 on_begin() 、执行后回调 on_end()、执行异常则回调 on_exception() """
                # on_begin
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
                return _meter

            return _newfunc

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
            super().__setattr__(name, value)
            return

        # if name in self.__dict__: TODO workaround multi-gpu error: Expected to mark a variable ready only once
        #     self.__dict__.pop(name)

        self._state_dicts.setdefault(type_name, set()).add(name)
        self._rev_index[name] = type_name

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
        file = inspect.getfile(cls)
        return os.path.basename(os.path.dirname(file))

    @classmethod
    def filebasename(cls):
        try:
            file = inspect.getfile(cls)
            pre = os.path.splitext(os.path.basename(file))[0]
        except:
            pre = 'builtin'
        return pre

    @classmethod
    def generate_exp_name(cls) -> str:
        pre = cls.filebasename()

        exp_name = cls.__exp_name__
        if exp_name is None:
            exp_name = cls.__name__.lower().replace("trainer", "exp")

        return "{}.{}".format(pre.lower(), exp_name.lower())

    def on_trainer_exception(self, func: Callable, exception: BaseException):
        pass
