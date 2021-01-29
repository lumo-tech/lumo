"""

"""
import bisect
import os
import pprint as pp
import warnings

# from ..utils.lazy import torch, np
import numpy as np
import torch
from functools import lru_cache
from functools import wraps
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from typing import Any, Union, List, Dict
from collections.abc import Iterator
from .databundler import DataBundler
from .meter import AvgMeter, Meter
from .params import Params
from .saver import Saver
from ..base_classes.metaclasses import Merge
from ..globals import _BUILTIN_PLUGIN, _FNAME, _PLUGIN_DIRNAME, _PLUGIN_KEY, _OS_ENV


def mp_agent(rank, self, op):
    import torch.distributed as dist
    self.params.local_rank = rank
    dist.init_process_group(backend='nccl', init_method=self.params.init_method,
                            rank=rank,
                            world_size=self.params.world_size)

    self.params.device = 'cuda:{}'.format(rank)
    self.regist_device(torch.device(self.params.device))
    torch.cuda.set_device(self.params.local_rank)
    print('in rank {}'.format(rank))
    self.models(self.params)
    self.datasets(self.params)
    self.callbacks(self.params)
    op(self)


class BaseTrainer(metaclass=Merge):
    __exp_name__ = None
    _call_backs = {
        "initial",
        "train", "train_epoch", "train_step", "test", "eval", "train_on_batch",
        "regist_databundler", "train_batch", "test_eval_logic", "test_eval_logic_v2", "predict",
        "load_keypoint", "load_checkpoint", "load_model", "save_keypoint", "save_checkpoint", "save_model",
    }

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if cls.__exp_name__ is None:
            cls.__exp_name__ = cls.__name__.lower().replace("trainer", "Exp")

        def wrapper(func, _call_set: list):
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
                    # 如果返回迭代器，那么会消耗掉迭代器，并只返回最后一次运行的结果（如果有）
                    if isinstance(_meter, Iterator):
                        _m = Meter()
                        for _m in _meter:
                            pass
                        _meter = _m
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

        self._callback_set = []
        self._callback_name_set = set()

        vars = dir(self)
        for name in vars:
            if name not in self._call_backs:
                continue
            if name.startswith("_"):
                continue
            value = getattr(self, name, None)
            if value is None:
                continue
            if callable(value):
                setattr(self, name, wrapper(value, self._callback_set))
        return self

    def __init__(self, params: Params = None):
        self._model_dict = {}  # type:Dict[str,torch.nn.Module]
        self._optim_dict = {}  # type:Dict[str,Optimizer]
        self._other_state_dict = {}
        self._vector_dict = {}
        self._checkpoint_plug = {}
        self._databundler_dict = {}  # type:Dict[str,DataBundler]
        self.train_epoch_toggle = False
        self.train_toggle = False
        self.experiment = None
        if params is not None:
            self.params = params
            if isinstance(params.device, str):
                _device = torch.device(params.device)
                self.regist_device(_device)
                if 'cuda' in params.device:
                    torch.cuda.set_device(_device)
            else:
                assert False
            # elif isinstance(params.device, (list, dict)):
            #     if isinstance(params.device, list):
            #         self.regist_devices([torch.device(i) for i in params.device])
            #     elif isinstance(params.device, dict):
            #         self.regist_devices({k: torch.device(v) for k, v in params.device.items()})
            #     elif isinstance(params.device, torch.device):
            #         warnings.warn("define torch.device in params is not recommanded, allocate a string is better.")
            #         self.regist_device(params.device)
            #     else:
            #         warnings.warn("Unknown type for params.device.")

            if params.contains('tmp_dir'):
                if params.tmp_dir is not None:
                    os.environ['TMPDIR'] = params.tmp_dir

            if params.local_rank >= 1:
                from thexp import globs
                globs['rank'] = params.local_rank
        else:
            self.params = Params()
        self.initial()

    def __setstate__(self, state):
        self._model_dict = state['_model_dict']
        self._optim_dict = state['_optim_dict']
        self._other_state_dict = state['_other_state_dict']
        self._vector_dict = state['_vector_dict']
        self._checkpoint_plug = state['_checkpoint_plug']
        self._databundler_dict = state['_databundler_dict']
        self.train_epoch_toggle = state['train_epoch_toggle']
        self.train_toggle = state['train_toggle']
        self.params = state['params']
        self.experiment = state.get('experiment', None)

    def __getstate__(self):
        res = {
            '_model_dict': self._model_dict,
            '_optim_dict': self._optim_dict,
            '_other_state_dict': self._other_state_dict,
            '_vector_dict': self._vector_dict,
            '_checkpoint_plug': self._checkpoint_plug,
            '_databundler_dict': self._databundler_dict,
            'train_epoch_toggle': self.train_epoch_toggle,
            'train_toggle': self.train_toggle,
            'experiment': self.experiment,
            'params': self.params,
        }
        empk = [k for k in res if res[k] is None]
        for k in empk:
            res.pop(k)
        return res

    def initial(self):
        """initial the trainer"""
        import inspect
        from .experiment import Experiment
        # build experiment
        self.params.initial()
        file = inspect.getfile(self.__class__)
        dirname = os.path.basename(os.path.dirname(file))

        pre = os.path.splitext(os.path.basename(file))[0]

        if not self.params.get('git_commit', True):
            os.environ[_OS_ENV.THEXP_COMMIT_DISABLE] = '1'

        self.experiment = Experiment("{}.{}".format(pre, dirname))

        # rigist and save params of this training procedure
        self.experiment.add_params(self.params)

        # regist trainer info
        trainer_kwargs = {
            _PLUGIN_KEY.TRAINER.path: inspect.getfile(self.__class__),
            _PLUGIN_KEY.TRAINER.doc: self.__class__.__doc__,
            _PLUGIN_KEY.TRAINER.fn: pre,
            _PLUGIN_KEY.TRAINER.class_name: self.__class__.__name__
        }
        self.experiment.add_plugin(_BUILTIN_PLUGIN.trainer, trainer_kwargs)

        self.callbacks(self.params)
        self.models(self.params)
        self.datasets(self.params)

    def _regist_databundler(self, key, val):
        from torch.utils.data import DataLoader
        assert isinstance(val, (DataBundler, DataLoader))
        if isinstance(val, DataLoader):
            val = DataBundler().add(val)

        # To ensure that children threads(dataloader workers)  will be killed
        if key in self._databundler_dict:
            del self._databundler_dict[key]

        self._databundler_dict[key] = val

    def regist_device(self, device: torch.device):
        self.device = device

    def regist_databundler(self,
                           train: Union[DataBundler, DataLoader] = None,
                           eval: Union[DataBundler, DataLoader] = None,
                           test: Union[DataBundler, DataLoader] = None):
        """
        regist train/eval/test dataloader

        Args:
            train / eval / test: DataBundler in thexp, or DataLoader in pytorch
                if None, the corresponding train/eval/test methods will be ignored when call them.
        """
        if train is not None:
            self._regist_databundler("train", train)
        if eval is not None:
            self._regist_databundler("eval", eval)
        if test is not None:
            self._regist_databundler("tests", test)

        self.logger.info(self._databundler_dict)

    def stop_train(self):
        """stop current training procedure"""
        self.train_toggle = True

    def stop_current_epoch(self):
        """stop current training epoch, if current epoch doesn't reach the end,
        the next epoch will be started and the releated callback methods will be called.
        """
        self.train_epoch_toggle = True

    def train(self):
        params = self.params
        while params.eidx < params.epoch + 1:
            self.train_epoch(params.eidx, params)
            params.eidx += 1
            if self.train_toggle:
                self.train_toggle = False
                break

    def train_epoch(self, eidx: int, params: Params):
        avg = AvgMeter()
        self.change_mode(True)
        for idx, batch_data in enumerate(self.train_dataloader):  # 复现多线程下 Keyboard Interupt，尝试通过Try解决
            meter = self.train_batch(eidx, idx, self.params.global_step, batch_data, params, self.device)
            avg.update(meter)
            # del meter

            params.global_step += 1
            params.idx = idx
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

        self.change_mode(False)
        return avg

    def train_step(self, steps) -> Union[AvgMeter, Meter]:
        """
        train specific steps
        Args:
            steps: int

        Returns:
            if steps > 1, will return a AveMeter instance.
            if steps = 1, will return a Meter object.

        """
        param = self.params
        i = 0
        if steps == 1:
            for idx, data in enumerate(self.train_dataloader):
                return self.train_batch(0, idx, i, data, param, self.device)

        avg = AvgMeter()
        while steps > 0:
            avg = AvgMeter()
            for idx, data in enumerate(self.train_dataloader):
                meter = self.train_batch(0, idx, i, data, param, self.device)
                steps -= 1
                avg.update(meter)
                if steps <= 0:
                    return avg
        return avg

    def feed_batchdata(self, batch_data=None) -> Meter:
        """
        train a step for testing or some other purpose.
        Args:
            batch_data: the batch_data used in training procedure.
                if None, batch_data will be fetched from train_datasetloader.

        Returns:
            a Meter.
        """
        if batch_data is None:
            return self.train_step(1)
        return self.train_batch(0, 0, 0, batch_data, self.params, self.device)

    def test(self):
        """test via test_dataloader"""

        loader = self.test_dataloader
        if loader is None:
            self.logger.info("Have no test dataset, ignored test.")
            return None
        return self.test_eval_logic_v2(loader, self.params, True)

    def eval(self):
        """eval via eval_dataloader"""
        loader = self.eval_dataloader
        if loader is None:
            self.logger.info("Have no eval dataset, ignored eval.")
            return None
        return self.test_eval_logic_v2(loader, self.params, False)

    @property
    def in_main_process(self):
        return self.params.local_rank <= 0

    @property
    def safe_writer(self):
        """see trainer.writer"""
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        return self.writer

    @property
    @lru_cache()
    def writer(self):
        """
        Notes:
        ------
        When using add_embedding, there may raise some exceptions cased by version conflict, here is some solutions:

        1. tensorflow_core._api.v1.io.gfile' or'tensorflow_core._api.v2.io.gfile' has no attribute 'get_filesystem'
        first, try upgrade tensorboard and tensorflow as followed version:
            tensorboard==2.0.2
            tensorflow==2.0.0

        if you still have the same problem, use this code as a temporary solution:

            import tensorflow as tf
            import tensorboard as tb
            tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

        use `trainer.safe_writer` to get a writter with these code added inside thexp.

        solution is referred by https://github.com/pytorch/pytorch/issues/30966


        2. You may cause PermissionError like: [Errno 13] Permission denied: '/tmp/.tensorboard-info/pid-20281.info'
        the solution is to set environment variable TMPDIR

            export TMPDIR=/tmp/$USER;
            mkdir -p $TMPDIR;
            tensorboard --logdir ...

        code in line:
            export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir ....

        solution is referred by https://github.com/tensorflow/tensorboard/issues/2010

        Returns:
            A SummaryWriter instance
        """
        from torch.utils.tensorboard import SummaryWriter

        kwargs = self.experiment.add_board()
        res = SummaryWriter(**kwargs)

        def close(*args):
            res.flush()
            res.close()

        self.experiment.regist_exit_hook(close)
        return res

    @property
    @lru_cache()
    def logger(self):
        """see thexp.frame.Logger"""
        from .logger import Logger
        logger = Logger()
        fn = logger.add_log_dir(self.experiment.test_dir)
        self.experiment.add_logger(fn)
        return logger

    @property
    @lru_cache()
    def saver(self):
        """see thexp.frame.Saver"""
        kwargs = self.experiment.add_saver()
        return Saver(**kwargs)

    @property
    @lru_cache()
    def rnd(self):
        """see thexp.frame.RndManager"""
        from .rndmanager import RndManager
        kwargs = self.experiment.add_rndmanager()
        return RndManager(**kwargs)

    @property
    def model_dict(self) -> Dict[str, torch.nn.Module]:
        return self._model_dict

    @property
    def optimizer_dict(self) -> Dict[str, Optimizer]:
        return self._optim_dict

    @property
    def train_dataloader(self) -> DataBundler:
        return self._databundler_dict.get("train", None)

    @property
    def eval_dataloader(self) -> DataBundler:
        return self._databundler_dict.get("eval", None)

    @property
    def test_dataloader(self) -> DataBundler:
        return self._databundler_dict.get("tests", None)

    @classmethod
    def from_params(cls, params: Params = None):
        return cls(params)

    def regist_checkpoint(self, key, func):
        """
        注册需要被 checkpoint 类型的字典保存的
        Args:
            key:
            func:

        Returns:

        """
        self._checkpoint_plug[key] = func

    def save_keypoint(self, extra_info=None, replacement=False):
        """
        保存 keypoint，会保存所有可存储格式
        Args:
            extra_info:  额外的信息，将以 json 格式被保存在和模型文件名相同，但后缀名为 json 的文件中
            replacement: 若遇到相同文件名，是否进行替换

        Returns:

        """
        state_dict = self.checkpoint_dict()
        fn = self.saver.save_keypoint(self.params.eidx, state_dict, extra_info, replacement)
        self.logger.info("save keypoint in {}".format(fn))
        return fn

    def save_checkpoint(self, extra_info=None, replacement=False):
        """
        保存 checkpoint，会保存所有可存储格式
        Args:
            extra_info:  额外的信息，将以 json 格式被保存在和模型文件名相同，但后缀名为 json 的文件中
            replacement: 若遇到相同文件名，是否进行替换

        Returns:
            保存的 checkpoint 的文件名
        """
        state_dict = self.checkpoint_dict()
        fn = self.saver.save_checkpoint(self.params.eidx, state_dict, extra_info, replacement)
        self.logger.info("save checkpoint in {}".format(fn))
        return fn

    def save_model(self, extra_info: dict = None) -> str:
        """
        保存 model，只会保存所有 torch.nn.Module 类型的 state_dict
        Args:
            extra_info: 额外的信息，将以 json 格式被保存在和模型文件名相同，但后缀名为 json 的文件中

        Returns:
            所保存模型的文件名

        """
        state_dict = self.model_state_dict()
        fn = self.saver.save_model(self.params.eidx, state_dict, extra_info)
        self.logger.info("save model in {}".format(fn))
        return fn

    def add_callback(self, callback):
        """
        添加一个回调函数，注意，不能添加重复的 callback，这不推荐，也没有必要。
        :type callable,str
        :param callback:
        :return:
        """
        msg = None
        cb_name = callback.__class__.__name__
        if callback not in self._callback_set and cb_name in self._callback_name_set:
            msg = "Callback duplicate."
            callback.on_hook_failed(self, msg)

        if msg is not None:
            return False
        bisect.insort(self._callback_set, callback)
        self._callback_name_set.add(cb_name)

        callback._trainer = self
        callback.on_hooked(self, self.params)
        self.logger.info("{} hooked on {}.".format(callback, self))
        return True

    def reload_callback(self, callback):
        """重新加载某 callback"""
        self.remove_callback(callback.__class__)
        return self.add_callback(callback)

    def remove_callback(self, callback):
        """
        移除已加载的 callback
        Args:
            callback: 可以是回调类名、实例、或回调类类型

        Returns:
            是否移除成功，若返回False，则表明没有找到对应的 callback
            若返回 True，则表明该 callback 已被完好移除
        """
        msg = None
        from .callbacks import BaseCallback

        try:
            if issubclass(callback, BaseCallback):
                for cb in self._callback_set:
                    if cb.__class__.__name__ == callback.__name__:
                        callback = cb
                        break
        except:  # handle TypeError: issubclass() arg 1 must be a class
            pass

        if isinstance(callback, str):
            for cb in self._callback_set:
                if cb.__class__.__name__ == callback:
                    callback = cb
                    break

        if callback not in self._callback_set:
            return False

        cb_name = callback.__class__.__name__
        self._callback_set.remove(callback)
        self._callback_name_set.remove(cb_name)
        self.logger.info("{} unhooked from {}.".format(callback, self))
        return True

    '''module和optim的一部分方法集成'''

    def load_checkpoint(self, fn):
        ckpt, info = self.saver.load_state_dict(fn)
        self.load_checkpoint_dict(ckpt)
        self.logger.raw(pp.pformat(info))

    def load_model(self, fn, strict=True):
        ckpt, info = self.saver.load_state_dict(fn)
        self.load_model_state_dict(ckpt, strict=strict)
        self.logger.raw(pp.pformat(info))

    def load_checkpoint_dict(self, state_dict):
        self.logger.raw("loading checkpoint")
        self.params.eidx = state_dict['eidx']
        self.params.idx = state_dict['idx']
        self.params.global_step = state_dict['global_step']
        self.load_model_state_dict(state_dict["model"])
        self.load_optim_state_dict(state_dict["optim"])
        self.load_other_state_dict(state_dict["other"])
        self.load_vector_dict(state_dict["vector"])
        self.load_extra_state_dict(state_dict['plug'])

    def load_model_state_dict(self, state_dict, strict=True):
        self.logger.inline("loading model: ", append=True)
        for k in self._model_dict:
            self.logger.raw(k)
            if k in state_dict:
                model = self._model_dict[k]
                if hasattr(model,"_load_state_dict"):
                    model._load_state_dict(state_dict[k], strict=strict)
                else:
                    model.load_state_dict(state_dict[k], strict=strict)
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_optim_state_dict(self, state_dict, strict=False):
        self.logger.inline("loading optimizers: ", append=True)
        for k in self._optim_dict:
            self.logger.raw(k)
            if k in state_dict:
                self._optim_dict[k].load_state_dict(state_dict[k])
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_other_state_dict(self, state_dict, strict=False):
        self.logger.inline("loading other: ", append=True)
        for k in self._other_state_dict:
            self.logger.raw(k)
            if k in state_dict:
                self._other_state_dict[k].load_state_dict(state_dict[k])
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_vector_dict(self, state_dict, strict=False):
        self.logger.inline("loading vectors: ", append=True)
        for k in self._vector_dict:
            self.logger.raw(k)
            if k in state_dict:
                self.__setattr__(k, state_dict[k])
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def extra_state_dict(self) -> dict:
        """
        nn.Module, Optimizer, numpy.ndarray, torch.Tensor, 以及其他包含 state_dict 接口的对象在保存checkpoint时候
        无需手动添加，会自动被存储，而除了这些之外，有其他需要在 checkpoint 中被保存的内容，可以通过该接口实现
        Returns:

        Notes:
            该方法和 load_extra_state_dict() 一一对应，两者需要同时实现
        """
        return {}

    def load_extra_state_dict(self, state_dict, strict=False):
        """
        nn.Module, Optimizer, numpy.ndarray, torch.Tensor, 以及其他包含 state_dict 接口的对象在保存checkpoint时候
        无需手动添加，会自动被存储，而除了这些之外，有其他需要在 checkpoint 中被保存的内容，可以通过该接口实现
        Returns:

        Args:
            state_dict:
            strict:

        Returns:

        Notes:
            该方法和 extra_state_dict() 一一对应，两者需要同时实现
        """
        pass

    def load_state_dict(self, strict=True, **kwargs):
        """
        传入键值对，对 model / optim / other / checkpoint_plug 分别进行检查尝试，若能匹配则调用相应的加载方法
        Args:
            strict:  是否严格匹配，针对 model 和 checkpoint_plug ，当存在无法匹配的键时，
            若该值为 True，则抛出异常，否则仅报一次警告
            **kwargs:  键值对

        Returns:

        """
        for k, v in kwargs.items():
            if k in self._model_dict:
                self._model_dict[k].load_state_dict(v, strict)
            elif k in self._optim_dict:
                self._optim_dict[k].load_state_dict(v)
            elif k in self._other_state_dict:
                self._other_state_dict[k].load_state_dict(v)
            elif k in self._vector_dict:
                self.__setattr__(k, v)
            elif k in self._checkpoint_plug:
                self._checkpoint_plug[k](self, v, strict)
            elif strict:
                raise KeyError(k)
            else:
                warnings.warn("{} not found in all state_dict".format(k))

    def estimate_memory(self):
        for _, v in self._model_dict.items():
            pass

    def checkpoint_dict(self):
        val = dict(
            model=self.model_state_dict(),
            optim=self.optim_state_dict(),
            other=self.other_state_dict(),
            vector=self.vector_state_dict(),
            plug=self.extra_state_dict(),
            eidx=self.params.eidx,
            idx=self.params.idx,
            global_step=self.params.global_step,
            test_name=self.experiment.test_name,
        )
        return val

    def model_state_dict(self):
        """所有继承自 nn.module 的类的 state_dict """
        return {k: v.state_dict() for k, v in self._model_dict.items()}

    def optim_state_dict(self):
        """所有继承自 Optimizer 的类的 state_dict"""
        return {k: v.state_dict() for k, v in self._optim_dict.items()}

    def other_state_dict(self):
        """所有 实现了 state_dict / load_state_dict 接口的"""
        return {k: v.state_dict() for k, v in self._other_state_dict.items()}

    def vector_state_dict(self):
        """所有 torch.Tensor 或 numpy.ndarray"""
        return {k: v for k, v in self._vector_dict.items()}

    def change_mode(self, train=True):
        for k, v in self._model_dict.items():
            if train:
                v.train()
            else:
                v.eval()

    def to(self, device):
        for k, v in self._model_dict.items():
            self.__setattr__(k, v.to(device))
        for k, v in self._databundler_dict.items():
            v.to(device)
        for k, v in self._vector_dict.items():
            if isinstance(v, torch.Tensor):
                self.__setattr__(k, v.to(device))

    '''magic functions'''

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        from torch.optim.optimizer import Optimizer
        if isinstance(value, torch.device):
            pass
        elif isinstance(value, torch.nn.Module):
            self._model_dict[name] = value
        elif isinstance(value, Optimizer):
            self._optim_dict[name] = value
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            self._vector_dict[name] = value
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            self._other_state_dict[name] = value

    def __setitem__(self, key: str, value: Any):
        self.__setattr__(key, value)

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        raise NotImplementedError()

    def test_eval_logic(self, dataloader, param: Params):
        raise NotImplementedError()

    def test_eval_logic_v2(self, dataloader, param: Params, is_test: bool):
        return self.test_eval_logic(dataloader, param)

    def predict(self, xs):
        raise NotImplementedError()

    def callbacks(self, params: Params):
        """初始化回调函数"""
        pass

    def datasets(self, params: Params):
        """初始化数据集"""
        pass

    def models(self, params: Params):
        """初始化模型"""
        pass


class Trainer(BaseTrainer):

    def callbacks(self, params: Params):
        pass

    def datasets(self, params: Params):
        pass

    def models(self, params: Params):
        pass

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        pass

    def extra_state_dict(self) -> dict:
        return super().extra_state_dict()

    def load_extra_state_dict(self, state_dict, strict=False):
        super().load_extra_state_dict(state_dict, strict)


class WrapTrainer(Trainer):

    def __init__(self, params: Params, train_dataloader, model_with_loss_fn, optimize, eval_dataloader=None,
                 test_dataloader=None):
        super().__init__(params)

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)


class DistributedTrainer():
    def __init__(self, trainer_cls, params: Params, op):
        self.trainer_cls = trainer_cls
        self.params = params
        self.op = op

    def run(self):
        trainer = self.trainer_cls.__new__(self.trainer_cls)  # type:Trainer
        trainer._model_dict = {}  # type:Dict[str,torch.nn.Module]
        trainer._optim_dict = {}  # type:Dict[str,Optimizer]
        trainer._other_state_dict = {}
        trainer._vector_dict = {}
        trainer._checkpoint_plug = {}
        trainer._databundler_dict = {}  # type:Dict[str,DataBundler]
        trainer.train_epoch_toggle = False
        trainer.train_toggle = False
        trainer.experiment = None

        params = self.params
        if self.params is not None:
            trainer.params = params
            if isinstance(params.device, str):
                trainer.regist_device(torch.device(params.device))
            else:
                assert False

            if params.contains('tmp_dir'):
                if params.tmp_dir is not None:
                    os.environ['TMPDIR'] = params.tmp_dir

            if params.local_rank >= 1:
                from thexp import globs
                globs['rank'] = params.local_rank
        else:
            trainer.params = Params()

        import inspect
        from .experiment import Experiment
        # build experiment
        trainer.params.initial()
        file = inspect.getfile(trainer.__class__)
        dirname = os.path.basename(os.path.dirname(file))

        pre = os.path.splitext(os.path.basename(file))[0]

        if not trainer.params.get('git_commit', True):
            os.environ[_OS_ENV.THEXP_COMMIT_DISABLE] = '1'

        trainer.experiment = Experiment("{}.{}".format(pre, dirname))

        # rigist and save params of this training procedure
        trainer.experiment.add_params(params)

        # regist trainer info
        trainer_kwargs = {
            _PLUGIN_KEY.TRAINER.path: inspect.getfile(trainer.__class__),
            _PLUGIN_KEY.TRAINER.doc: trainer.__class__.__doc__,
            _PLUGIN_KEY.TRAINER.fn: pre,
            _PLUGIN_KEY.TRAINER.class_name: trainer.__class__.__name__
        }
        trainer.experiment.add_plugin(_BUILTIN_PLUGIN.trainer, trainer_kwargs)

        params.distributed = True
        import torch.multiprocessing as mp
        if params.world_size == -1:
            params.world_size = torch.cuda.device_count()

        mp.spawn(mp_agent,
                 args=(trainer, self.op),
                 nprocs=trainer.params.world_size)
