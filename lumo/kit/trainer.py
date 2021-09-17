"""

"""
import subprocess
import time
import sys
import bisect
import inspect
import os
from dataclasses import dataclass
from functools import wraps, lru_cache
from typing import Any, Dict, Union, Optional, Tuple, Sequence, Mapping
import random

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from functools import lru_cache
from lumo.contrib.itertools import safe_cycle
from .environ import globs
from .experiment import TrainerExperiment
from .logger import Logger
from .saver import Saver
from .meter import Meter, AvgMeter
from .rnd import RndManager
from .datamodule import DataModule
from .bundler import DataBundler
from .mixin import ModelMix, CallbackMix, DataModuleMix

from .params import ParamsType
from lumo.base_classes import attr, TrainerStage
from lumo.base_classes.metaclasses import Merge
from ..proc.const import TRAINER
from lumo.utils import dist
from lumo.utils.device import construct_device_args_kwargs, to_device_enumrate
from lumo.contrib.accelerate import Accelerator


@dataclass()
class initial_tuple():
    models: bool = False
    callbacks: bool = False
    optims: bool = False
    train_dataloader: bool = False
    val_dataloader: bool = False
    test_dataloader: bool = False


class TrainerResult(attr):
    MSG_OK = 'success'
    MSG_OK_MID = 'success stop in midway'
    MSG_NO_DATALOADER = 'dataloader not loaded'

    def __init__(self, state: TrainerStage, meter: AvgMeter, msg=''):
        super().__init__()
        self.state = state
        self.meter = meter
        self.msg = msg


def mp_agent(rank, trainer, op, dataloader):
    import torch.distributed as dist
    trainer.params.local_rank = rank
    dist.init_process_group(backend='nccl', init_method=trainer.params.init_method,
                            rank=rank,
                            world_size=trainer.params.world_size)

    trainer.params.device = 'cuda:{}'.format(rank)
    trainer.regist_device(torch.device(trainer.params.device))
    torch.cuda.set_device(trainer.params.local_rank)
    print('in rank {}'.format(rank))
    op(dataloader)


class _BaseTrainer(ModelMix, CallbackMix, metaclass=Merge):
    """
    1. 组装所有的插件
    2. 提供完整的训练流程 api
    3. 提供对需要 api 函数的 callback


    Trainer helps you to control all stage happened in deeplearning, including train/evaluation/test, save/load checkpoints
    , log/meter/tensorboard variable, etc. It's also very convenient to custom your own logits.

    Thanks for git, trainer will commit all change on `experiments` branch before you start your training, and
    this makes you easily



    When create a Trainer instance, a Params instance should be passed
    """

    __exp_name__ = None
    _call_backs = {
        "save_keypoint", "save_checkpoint", "save_model", "load_state_dict",
        'ioptims', 'icallbacks', 'imodels',
    }

    def _exit_hook(self, exc_type, exc, tb, *args):
        import traceback
        res = traceback.format_exception(exc_type, exc, tb)
        res = [i for i in res if 'in _newfunc' not in i]
        print(''.join(res), file=sys.stderr)

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if cls.__exp_name__ is None:
            cls.__exp_name__ = cls.__name__.lower().replace("trainer", "Exp")

        from lumo.utils.exithook import replace

        replace(self._exit_hook)

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

    @classmethod
    def _gene_class_exp_name(trainer_instance) -> str:
        try:
            file = inspect.getfile(trainer_instance)
            pre = os.path.splitext(os.path.basename(file))[0]
        except:
            pre = 'builtin'

        return "{}.{}".format(pre.lower(), trainer_instance.__exp_name__.lower())

    def __init__(self, params: ParamsType):
        self._state_dicts = attr({
            "models": {},
            "optims": {},
            "buffer": {
                'np': {},
                'th': {}
            },
            "others": {},
            'device': torch.device('cpu'),
            'params': params,
            'exp': TrainerExperiment(self._gene_class_exp_name()),
        })
        self._initialize_globs()
        self._initial_exp()

        self._logger = None
        self._rnd = None
        self._accelerate = None
        self._saver = None
        self._datamodule = None

        self.initial = initial_tuple()
        self.train_epoch_toggle = False
        self.train_toggle = False

        if 'device' in params:
            _device = torch.device(params.device)
            self.regist_device(_device)

        self._check_cb_init()
        self.exp.start()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        from torch.optim.optimizer import Optimizer

        if name.startswith('_') or name.endswith('_'):
            # when callback is trainer itself, _hooked will be passed, and caused recursion
            # if some pretrained models need to be ignored in save/load stage, it can be named like 'some_' or '_some'
            return

        if isinstance(value, torch.device):
            self._state_dicts[TRAINER.DKEY.devices][name] = value
        elif isinstance(value, torch.nn.Module):
            self._state_dicts[TRAINER.DKEY.models][name] = value
        elif isinstance(value, Optimizer):
            self._state_dicts[TRAINER.DKEY.optims][name] = value
        elif isinstance(value, (torch.Tensor)):
            self._state_dicts[TRAINER.DKEY.tensor]['th'][name] = value
        elif isinstance(value, (np.ndarray)):
            self._state_dicts[TRAINER.DKEY.tensor]['np'][name] = value
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            self._state_dicts[TRAINER.DKEY.others][name] = value

    def __setitem__(self, key: str, value: Any):
        self.__setattr__(key, value)

    def __setstate__(self, state):
        self._state_dicts = attr.from_dict(state)

    def __getstate__(self):
        res = self._state_dicts.pickify()
        return res

    def _initialize_globs(self):
        globs['rank'] = dist.local_rank()
        globs['world_size'] = dist.world_size()
        globs['LOCAL_RANK'] = globs['rank']
        globs['WORLD_SIZE'] = dist.world_size()

        for k, v in self.params.items():  # type:str,Any
            if k.isupper():
                globs[k] = v
                if isinstance(v, str):
                    os.environ[k] = v
        for k in {'nocommit'}:
            if k in self.params:
                globs[k] = self.params[k]

    def _initial_exp(self):
        self.params.to_json(self.exp.params_fn)

    @staticmethod
    def _iter_dataloader(dataloader, device):
        pass

    def _check_optim_init(self):
        if not self.initial.optims:
            self.ioptims(self.params)
            self.initial.optims = True

    def _check_models_init(self):
        if not self.initial.models:
            self.imodels(self.params)
            self.initial.models = True

    def _check_cb_init(self):
        if not self.initial.callbacks:
            self.icallbacks(self.params)
            self.initial.callbacks = True

    def get_state(self, key, default=None):
        return self._state_dicts.get(key, default)

    def regist_state(self, key, value=None, picklable=True):
        if picklable:
            self._state_dicts[key] = value

    @property
    def is_debug(self):
        return self.params.get('debug', False)

    @property
    def global_step(self):
        return self.params.global_step

    @property
    def eidx(self):
        return self.params.eidx

    @property
    def is_dist(self):
        return dist.is_dist()

    @property
    def local_rank(self):
        return dist.local_rank()

    @property
    def world_size(self):
        return dist.world_size()

    @property
    def exp(self) -> TrainerExperiment:
        return self._state_dicts[TRAINER.DKEY.exp]

    @property
    def logger(self):
        if self._logger is None:
            from lumo.kit.logger import set_global_logger
            self._logger = Logger()
            set_global_logger(self._logger)
            if self.params.get('debug', False):
                self._logger.set_verbose(Logger.V_DEBUG)
                self._logger.dddebug('Enable debug log.')
            fn = self._logger.add_log_dir(self.exp.log_dir)
            self.exp.writeline('logger', fn)
        return self._logger

    @property
    def params(self) -> ParamsType:
        return self._state_dicts[TRAINER.DKEY.params]

    @property
    def accelerator(self) -> Accelerator:
        if self._accelerate is None:
            acce = {'device_placement': True,
                    'split_batches': False,
                    'fp16': None,
                    'cpu': False,
                    'rng_types': None,
                    'kwargs_handlers': None,
                    'device': self.params.get('device', None)}
            for k in acce:
                if k in self.params:
                    acce[k] = self.params[k]

            self._accelerate = Accelerator(**acce)
        return self._accelerate

    @property
    def rnd(self) -> RndManager:
        if self._rnd is None:
            self._rnd = RndManager(self.exp.rnd_dir)
        return self._rnd

    @property
    def saver(self) -> Saver:
        if self._saver is None:
            self._saver = Saver(self.exp.saver_dir)
        return self._saver

    @property
    def device(self) -> torch.device:
        return self._state_dicts[TRAINER.DKEY.device]

    @property
    def device_arg_kwargs(self) -> Tuple[Sequence, dict]:
        return construct_device_args_kwargs(self.device)

    @property
    def _models(self):
        return self._state_dicts[TRAINER.DKEY.models]

    @property
    def _optims(self):
        return self._state_dicts[TRAINER.DKEY.optims]

    @property
    def model_dict(self) -> Dict[str, nn.Module]:
        self._check_models_init()
        return self._models

    @property
    def optim_dict(self) -> Dict[str, Optimizer]:
        self._check_optim_init()
        return self._state_dicts[TRAINER.DKEY.optims]

    @property
    def buffers(self) -> Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]:
        return self._state_dicts[TRAINER.DKEY.tensor]

    @property
    def nparr(self) -> Dict[str, np.ndarray]:
        return self._state_dicts[TRAINER.DKEY.tensor]['np']

    @property
    def tharr(self) -> Dict[str, torch.Tensor]:
        return self._state_dicts[TRAINER.DKEY.tensor]['th']

    @property
    def others(self) -> Dict[str, Any]:
        return self._state_dicts[TRAINER.DKEY.others]

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

        kwargs = self.exp.board_args
        res = SummaryWriter(**kwargs)

        def close(*args):
            res.flush()
            res.close()

        self.exp.add_exit_hook(close)
        return res

    @property
    def datamodule(self):
        if self._datamodule is None:
            self._datamodule = DataModule()
        return self._datamodule

    @property
    def train_dataloader(self):
        return self.datamodule.train_dataloader

    @property
    def val_dataloader(self):
        return self.datamodule.val_dataloader

    @property
    def test_dataloader(self):
        return self.datamodule.test_dataloader

    def regist_device(self, device: torch.device):
        self._state_dicts[TRAINER.DKEY.device] = device
        if device.type == 'cuda':
            torch.cuda.set_device(device)

    def regist_dataloader(self, train: DataLoader = None, val: DataLoader = None, test: DataLoader = None,
                          datamodule: DataModuleMix = None):
        if datamodule is not None:
            self._datamodule = datamodule
        else:
            self.datamodule.regist_dataloader(train=train, val=val, test=test)

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
            return

        if callback.only_main_process and self.local_rank > 0:
            msg = f"{callback.__class__.__name__} only_main_process but in local_rank {self.local_rank}"
            callback.on_hook_failed(self, msg)
            return

        if callback.only_single_gpu and self.is_dist:
            msg = f"{callback.__class__.__name__} only_single_gpu but dist={self.is_dist}"
            callback.on_hook_failed(self, msg)
            return

        if msg is not None:
            return False
        bisect.insort(self._callback_set, callback)
        self._callback_name_set.add(cb_name)

        callback._hooked = self
        callback.on_hooked(self, self.params)
        self.logger.info("{} hooked on {}().".format(callback, self.__class__.__name__))
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

    def change_mode(self, train=True):
        for k, v in self.model_dict.items():
            if train:
                v.train()
            else:
                v.eval()

    def optim_state_dict(self):
        return {k: v.state_dict() for k, v in self.optim_dict.items()}

    def model_state_dict(self):
        return {k: v.state_dict() for k, v in self.model_dict.items()}

    def buffer_state_dict(self):
        return self.buffers

    def other_state_dict(self):
        return {k: v.state_dict() for k, v in self.others.items()}

    def state_dict(self):
        res = {
            'models': self.model_state_dict(),
            'optims': self.optim_state_dict(),
            'other': self.other_state_dict(),
            'buffer': self.buffer_state_dict(),
        }
        for k, v in self._state_dicts.items():
            if k not in res:
                res[k] = v

        return res

    def _load_fun_state_dict(self, src: dict, tgt: dict):
        for k, v in tgt.items():
            if k in src:
                v.load_state_dict(src[k])

    def ignore_object_state_dict(self, item):
        pass

    def load_state_dict(self, object: Union[str, dict]):
        if isinstance(object, str):
            ckpt, meta_info = self.saver.load_state_dict(object, with_meta=True)
        else:
            ckpt, meta_info = object, None

        _sub = {'models', 'optims', 'other'}
        if ckpt is None:
            self.logger.error('state_dict object is None, `load_state_dict()` ignored.')
            return meta_info

        for k, v in ckpt.items():
            if k in _sub:
                self._load_fun_state_dict(v, self._state_dicts[k])
            else:
                self._state_dicts[k] = v
        return meta_info

    def _build_trainer_meta_info(self, meta_info: Union[str, dict, Meter] = None):
        info = Meter()
        info['eidx'] = self.eidx
        if meta_info is not None:
            if isinstance(meta_info, str):
                info['msg'] = meta_info
            if isinstance(meta_info, Meter):
                meta_info = meta_info.serialize()
            if isinstance(meta_info, dict):
                info.update(meta_info)
        return info.serialize()

    def save_checkpoint(self, max_keep=10, is_best=False, meta_info: Union[str, dict, Meter] = None):
        info = self._build_trainer_meta_info(meta_info)
        return self.saver.save_checkpoint(self.eidx, self.state_dict(),
                                          meta_info=info,
                                          max_keep=max_keep,
                                          is_best=is_best)

    def save_keypoint(self, meta_info: Union[str, dict, Meter] = None):
        info = self._build_trainer_meta_info(meta_info)
        return self.saver.save_keypoint(self.eidx, self.state_dict(),
                                        meta_info=info)

    def save_model(self, is_best=False, meta_info: Union[str, dict, Meter] = None):
        info = self._build_trainer_meta_info(meta_info)
        return self.saver.save_model(self.eidx, {'models': self.model_state_dict()},
                                     meta_info=info,
                                     is_best=is_best)


class DLLoopMix():
    def stop_train(self):
        raise NotImplementedError()

    def stop_train_epoch(self):
        raise NotImplementedError()

    def to_stage(self, stage: TrainerStage):
        raise NotImplementedError()

    def train(self, dataloader: Union[DataLoader, DataModule]):
        raise NotImplementedError()

    def train_epoch(self, dataloader: DataLoader):
        raise NotImplementedError()

    def train_step(self, idx, batch, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, dataloader: Union[DataLoader, DataModule]):
        raise NotImplementedError()

    def evaluate_step(self, idx, batch, *args, **kwargs):
        raise NotImplementedError()

    def test(self, dataloader: Union[DataLoader, DataModule]):
        raise NotImplementedError()

    def test_step(self, idx, batch, *args, **kwargs):
        raise NotImplementedError()

    def inference(self, batch):
        raise NotImplementedError()

    def predict(self, batch):
        """alias of inference"""
        raise NotImplementedError()


class Trainer(DLLoopMix, _BaseTrainer):
    _call_backs = {
        "train", "train_epoch", "train_step",
        "test", "test_step",
        "evaluate", "evaluate_step",
        "predict", "inference",
        'prepare_dataloader',
    }

    def _wrap_result(self, meter: Union[Mapping, Meter, Sequence, torch.Tensor, np.ndarray]) -> dict:
        if meter is None:
            return {}
        if isinstance(meter, (Mapping, Meter)):
            return meter
        elif isinstance(meter, Sequence):
            return {f'm{i}': v for i, v in enumerate(meter)}
        elif isinstance(meter, (torch.Tensor, np.ndarray)):
            return {'metric': meter}

    @property
    def training(self):
        return self.params.stage == TrainerStage.train.name

    def stop_train(self):
        self.train_toggle = True

    def stop_train_epoch(self):
        self.train_epoch_toggle = True

    def to_stage(self, stage: TrainerStage):
        self.params.stage = stage.name
        self.change_mode(stage.name == TrainerStage.train.name)

    def to_device(self, item: Optional[Union[nn.Module, torch.Tensor, Sequence, Mapping]] = None,
                  device_args_kwargs=None):
        if device_args_kwargs is None:
            device_args_kwargs = self.device_arg_kwargs
        from lumo.utils.device import to_device
        if item is None:
            for k in self._models:
                v = self._models[k]
                if not self.accelerator.device_placement:
                    v.to(self.accelerator.device)
                v = self.accelerator.prepare(self._models[k])
                self._models[k] = v
            for k in self._optims:
                v = self.accelerator.prepare(self._optims[k])
                self._optims[k] = v
        else:
            return to_device(item, device_args_kwargs)

    def prepare_dataloader(self, stage: TrainerStage, dataloader=None) -> DataLoader:
        """
        prepare DataLoader instance for next trainer stage(train/test/evaluate)
        Args:
            stage:
            dataloader:

        Returns:

        """
        params = self.params
        initialized = getattr(self.initial, f"{stage.name}_dataloader")

        if dataloader is not None:
            dataloader_ = None
            """use loader from passed params first, then use itself, finally"""
            if isinstance(dataloader, DataModuleMix):
                dataloader.iidataloader(params, stage, initialized)
                self.regist_dataloader(datamodule=dataloader)
                dataloader_ = getattr(dataloader, f'{stage.name}_dataloader', None)
            elif isinstance(dataloader, (DataLoader, DataBundler)):
                self.regist_dataloader(**{stage.name: dataloader})
                dataloader_ = dataloader
        elif isinstance(self, DataModuleMix):
            self.iidataloader(params, stage, initialized)
            dataloader_ = getattr(self, f'{stage.name}_dataloader', None)

        if dataloader is None:
            if initialized:
                self.datamodule.iidataloader(params, stage, initialized)
            dataloader_ = getattr(self.datamodule, f'{stage.name}_dataloader', None)

        if isinstance(dataloader_, DataLoader):
            dataloader_ = self.accelerator.prepare(dataloader_)
        elif isinstance(dataloader_, DataBundler):
            for k, (loader, func) in list(dataloader_.dataloaders.items()):
                dataloader_.dataloaders[k] = [self.accelerator.prepare(loader), func]

        kwargs = {stage.name: dataloader_}
        self.regist_dataloader(**kwargs)
        return dataloader_

    def train(self, dataloader: Union[DataLoader, DataModuleMix] = None) -> TrainerResult:

        dataloader = self.prepare_dataloader(TrainerStage.train, dataloader)
        if dataloader is None:
            return TrainerResult(TrainerStage.train, 1, 'no train_dataloader')

        self._check_models_init()
        self._check_optim_init()

        self.initial.train_dataloader = True

        result = TrainerResult(TrainerStage.train, None, TrainerResult.MSG_OK)
        while self.params.eidx < self.params.epoch:
            self.params.eidx += 1
            self.to_stage(TrainerStage.train)
            result = self.train_epoch(dataloader)
            if self.train_toggle:
                self.train_toggle = False
                return TrainerResult(TrainerStage.train, result.meter, TrainerResult.MSG_OK_MID)

            self.exp.dump_train_info(self.params.eidx)

        return TrainerResult(TrainerStage.train, result.meter, TrainerResult.MSG_OK)

    def train_global_steps(self, steps=1, dataloader: Union[DataLoader, DataModuleMix] = None) -> TrainerResult:
        dataloader = self.prepare_dataloader(TrainerStage.train, dataloader)
        if dataloader is None:
            return TrainerResult(TrainerStage.train, 1, 'no train_dataloader')

        self._check_models_init()
        self._check_optim_init()

        self.initial.train_dataloader = True

        idx = 0
        loader = safe_cycle(self.train_dataloader)
        avg = AvgMeter()
        while idx < steps:
            self.params.global_step += 1
            self.params.idx = idx
            batch = next(loader)
            meter = self.train_step(idx, batch, self.params)
            avg.update(self._wrap_result(meter))
            idx += 1
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break
        return TrainerResult(TrainerStage.train_epoch, avg, TrainerResult.MSG_OK)

    def train_epoch(self, dataloader: DataLoader) -> TrainerResult:
        avg = None
        for idx, batch in enumerate(dataloader):
            if avg is None:
                avg = AvgMeter()
            self.params.global_step += 1
            self.params.idx = idx
            meter = self.train_step(idx, batch, self.params)
            avg.update(self._wrap_result(meter))
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break
        return TrainerResult(TrainerStage.train_epoch, avg, TrainerResult.MSG_OK)

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        pass

    def evaluate(self, dataloader: Union[DataLoader, DataModule] = None) -> TrainerResult:
        dataloader = self.prepare_dataloader(TrainerStage.val, dataloader)
        if dataloader is None:
            return TrainerResult(TrainerStage.val, None, TrainerResult.MSG_NO_DATALOADER)
        self._check_models_init()
        self.to_stage(TrainerStage.val)

        with torch.no_grad():
            avg = AvgMeter()
            for idx, batch in to_device_enumrate(dataloader, self.device_arg_kwargs):
                self.params.idx = idx
                meter = self.evaluate_step(idx, batch, self.params)
                avg.update(self._wrap_result(meter))

        return TrainerResult(TrainerStage.val, avg, TrainerResult.MSG_OK)

    def evaluate_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Meter:
        pass

    def test(self, dataloader: Union[DataLoader, DataModule] = None) -> TrainerResult:

        dataloader = self.prepare_dataloader(TrainerStage.test, dataloader)
        if dataloader is None:
            return TrainerResult(TrainerStage.test, None, TrainerResult.MSG_NO_DATALOADER)
        self._check_models_init()
        self.to_stage(TrainerStage.test)

        with torch.no_grad():
            avg = AvgMeter()
            for idx, batch in to_device_enumrate(dataloader, self.device_arg_kwargs):
                self.params.idx = idx
                meter = self.test_step(idx, batch, self.params)
                avg.update(self._wrap_result(meter))
        return TrainerResult(TrainerStage.test, avg, TrainerResult.MSG_OK)

    def test_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        pass

    def inference(self, batch):
        pass

    def predict(self, batch):
        """alias of inference"""
        self.inference(batch)

    def ioptims(self, params: ParamsType):
        pass

    def icallbacks(self, params: ParamsType):
        pass

    def imodels(self, params: ParamsType):
        pass
