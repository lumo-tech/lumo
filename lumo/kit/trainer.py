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

from .datamodule import DataModule
from .environ import globs
from .experiment import TrainerExperiment
from .logger import Logger
from .saver import Saver
from .meter import Meter, AvgMeter
from .rnd import RndManager
from .datamodule import DataModule
from .mixin import ModelMix, CallbackMix

from .params import DistributionParams, ParamsType
from lumo.base_classes import attr, TrainerStage
from lumo.base_classes.metaclasses import Merge
from lumo.utils.keys import TRAINER
from lumo.utils.connect import find_free_network_port
from lumo.utils.device import get_to_device_func, construct_device_args_kwargs
from ..base_classes.enums import TrainerStage


@dataclass()
class initial_tuple():
    models: bool = False
    callbacks: bool = False
    optims: bool = False
    train_dataloader: bool = False
    val_dataloader: bool = False
    test_dataloader: bool = False


class TrainerResult(attr):
    def __init__(self, state: TrainerStage, result: int, msg=''):
        super().__init__()
        self.state = state
        self.result = result
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


def to_device_enumrate(loader: DataLoader, device_args_kwargs: Tuple[Sequence, Dict]):
    to_device = get_to_device_func()
    for idx, batch in enumerate(loader):
        batch = to_device(batch, device_args_kwargs)
        yield idx, batch


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
        "initial",
        "train", "train_epoch", "train_step",
        "test", "test_step", "evaluate", "evaluate_step",
        "predict",
        "save_keypoint", "save_checkpoint", "save_model",
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
            'exp': TrainerExperiment(self._gene_class_exp_name()).start(),
        })
        self._initialize_globs()

        self._logger = None
        self._rnd = None
        self._saver = None
        self._datamodule = None

        self.initial = initial_tuple()
        self.train_epoch_toggle = False
        self.train_toggle = False

        if 'device' in params:
            _device = torch.device(params.device)
            self.regist_device(_device)

        self._check_cb_init()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        from torch.optim.optimizer import Optimizer
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
        globs['rank'] = -1
        globs['world_size'] = 0
        if dist.is_available():
            if dist.is_initialized():
                globs['rank'] = dist.get_rank()
                globs['world_size'] = dist.get_world_size()

        for k, v in self.params.items():  # type:str,Any
            if k.isupper():
                globs[k] = v
                if isinstance(v, str):
                    os.environ[k] = v

    @staticmethod
    def _iter_dataloader(dataloader, device):
        pass

    def _check_optim_init(self):
        if not self.initial.optims:
            self.ioptims(self.params)

    def _check_models_init(self):
        if not self.initial.models:
            self.imodels(self.params)

    def _check_cb_init(self):
        if not self.initial.callbacks:
            self.icallbacks(self.params)

    def get_state(self, key, default=None):
        return self._state_dicts.get(key, default)

    def regist_state(self, key, value=None, picklable=True):
        if picklable:
            self._state_dicts[key] = value

    @property
    def global_step(self):
        return self.params.global_step

    @property
    def eidx(self):
        return self.params.eidx

    @property
    def is_dist(self):
        return self.local_rank >= 0

    @property
    def local_rank(self):
        return globs['rank']

    @property
    def world_size(self):
        return globs['world_size']

    @property
    def exp(self) -> TrainerExperiment:
        return self._state_dicts[TRAINER.DKEY.exp]

    @property
    def logger(self):
        if self._logger is None:
            self._logger = Logger()
            self._logger.add_log_dir(self.exp.log_dir)
        return self._logger

    @property
    def params(self) -> ParamsType:
        return self._state_dicts[TRAINER.DKEY.params]

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
                          datamodule: DataModule = None):
        if datamodule is not None:
            self._datamodule = datamodule
        else:
            self._datamodule.regist_dataloader(train=train, val=val, test=test)

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

    def load_state_dict(self, state_dict: dict):
        _sub = {'models', 'optims', 'other'}
        for k, v in state_dict.items():
            if k in _sub:
                self._load_fun_state_dict(v, self._state_dicts[k])
            else:
                self._state_dicts[k] = v

    def _build_trainer_meta_info(self, meta_info: Union[str, dict] = None):
        info = {}
        info['eidx'] = self.eidx
        if meta_info is not None:
            if isinstance(meta_info, str):
                info['msg'] = meta_info
            if isinstance(meta_info, dict):
                info.update(meta_info)
        return info

    def save_checkpoint(self, max_keep=10, is_best=False, meta_info: Union[str, dict] = None):
        info = self._build_trainer_meta_info(meta_info)
        return self.saver.save_checkpoint(self.eidx, self.state_dict(),
                                          meta_info=info,
                                          max_keep=max_keep,
                                          is_best=is_best)

    def save_keypoint(self, meta_info: Union[str, dict] = None):
        info = self._build_trainer_meta_info(meta_info)
        return self.saver.save_keypoint(self.eidx, self.state_dict(),
                                        meta_info=info)

    def save_model(self, is_best=False, meta_info: Union[str, dict] = None):
        info = self._build_trainer_meta_info(meta_info)
        return self.saver.save_model(self.eidx, self.state_dict(),
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

    def evaluate_step(self, idx, batch, *args, **kwargs) -> Union[Dict, Meter]:
        raise NotImplementedError()

    def test(self, dataloader: Union[DataLoader, DataModule]):
        raise NotImplementedError()

    def test_step(self, idx, batch, *args, **kwargs) -> Union[Dict, Meter]:
        raise NotImplementedError()

    def inference(self, batch):
        raise NotImplementedError()

    def predict(self, batch):
        """alias of inference"""
        raise NotImplementedError()


class Trainer(DLLoopMix, _BaseTrainer):

    def _wrap_result(self, meter: Union[Mapping, Meter, Sequence, torch.Tensor, np.ndarray]):
        if meter is None:
            return {}
        if isinstance(meter, (Mapping, Meter)):
            return meter
        elif isinstance(meter, Sequence):
            return {f'm{i}': v for i, v in enumerate(meter)}
        elif isinstance(meter, (torch.Tensor, np.ndarray)):
            return {'metric': meter}

    def _check_dist_environ(self, loader: DataLoader):
        pass

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
        from lumo.utils.device import _to_device
        if item is None:
            for v in self._models.values():
                v.to(self.device)
        else:
            return _to_device(item, device_args_kwargs)

    def train(self, dataloader: Union[DataLoader, DataModule] = None):
        self.to_stage(TrainerStage.train)
        self.to_device()
        if dataloader is None:
            dataloader = self.train_dataloader
        elif isinstance(dataloader, DataModule):
            dataloader.idataloader(self.params, TrainerStage.train, self.initial.train_dataloader)
            self.initial.train_dataloader = True
            dataloader = dataloader.train_dataloader
        self.datamodule.regist_dataloader(train=dataloader)

        if dataloader is None:
            return TrainerResult(TrainerStage.train, 1, 'no train_dataloader')

        self._check_dist_environ(dataloader)
        self.ioptims(self.params)
        while self.params.eidx < self.params.epoch:
            self.train_epoch(dataloader)
            self.params.eidx += 1
            if self.train_toggle:
                self.train_toggle = False
                return TrainerResult(TrainerStage.train, 1, 'stoped midway')
            self.exp.dump_train_info(self.params.eidx)

        return TrainerResult(TrainerStage.train, 0)

    def train_epoch(self, dataloader: DataLoader):
        avg = AvgMeter()
        for idx, batch in to_device_enumrate(dataloader, self.device_arg_kwargs):
            meter = self.train_step(idx, batch, self.params)
            avg.update(self._wrap_result(meter))

            self.params.global_step += 1
            self.params.idx = idx
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break
        return avg

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        pass

    def evaluate(self, dataloader: Union[DataLoader, DataModule] = None):
        self.to_stage(TrainerStage.evaluate)
        if dataloader is None:
            dataloader = self.val_dataloader
        if isinstance(dataloader, DataModule):
            dataloader.idataloader(self.params, TrainerStage.evaluate, self.initial.val_dataloader)
            self.initial.val_dataloader = True
            dataloader = dataloader.val_dataloader
        if dataloader is None:
            return TrainerResult(TrainerStage.evaluate, 1, 'no eval_dataloader')

        self._check_dist_environ(dataloader)
        avg = AvgMeter()
        for idx, batch in to_device_enumrate(dataloader, self.device_arg_kwargs):
            meter = self.evaluate_step(idx, batch, self.params)
            avg.update(self._wrap_result(meter))
        return TrainerResult(TrainerStage.evaluate, 0)

    def evaluate_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Union[Dict, Meter]:
        return self.test_step(idx, batch, params, *args, **kwargs)

    def test(self, dataloader: Union[DataLoader, DataModule] = None):
        self.to_stage(TrainerStage.test)
        if dataloader is None:
            dataloader = self.test_dataloader
        if isinstance(dataloader, DataModule):
            dataloader.idataloader(self.params, TrainerStage.test, self.initial.test_dataloader)
            self.initial.test_dataloader = True
            dataloader = dataloader.test_dataloader
        if dataloader is None:
            return TrainerResult(TrainerStage.test, 1, 'no test_dataloader')

        self._check_dist_environ(dataloader)
        avg = AvgMeter()
        for idx, batch in to_device_enumrate(dataloader, self.device_arg_kwargs):
            meter = self.test_step(idx, batch, self.params)
            avg.update(self._wrap_result(meter))
        return TrainerResult(TrainerStage.test, 0)

    def test_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Union[Dict, Meter]:
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


class SingleMachineDDPSpawnDist(Trainer):
    pcls = DistributionParams

    def __init__(self, params: ParamsType, dist_params: DistributionParams):
        super().__init__(params)

        self.dist_params = params
        if params.world_size == -1:
            params.world_size = torch.cuda.device_count()

    def _check_dist_environ(self, loader: DataLoader):
        if self.is_dist:
            from torch.nn.parallel import DistributedDataParallel
            from torch.utils.data.distributed import DistributedSampler
            from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
            for k in self.model_dict:
                model = self.model_dict[k]
                if not isinstance(model, DistributedDataParallel):
                    model = DistributedDataParallel(model,
                                                    find_unused_parameters=True,
                                                    device_ids=[self.local_rank])
                self.model_dict[k] = model

            setattr(loader, f'_{loader.__class__.__name__}__initialized', False)
            if loader.batch_sampler is not None:  #
                sampler = loader.batch_sampler.sampler
            else:
                sampler = loader.sampler
            shuffle = False
            if isinstance(sampler, (RandomSampler, WeightedRandomSampler)):
                shuffle = True
            sampler = DistributedSampler(loader.dataset,
                                         shuffle=shuffle,
                                         drop_last=loader.drop_last)
            loader.sampler = sampler
            loader.persistent_workers = True
            setattr(loader, f'_{loader.__class__.__name__}__initialized', True)

    @property
    def num_nodes(self):
        return 1

    @property
    def world_size(self):
        return self.num_nodes * self.num_gpus

    @property
    def num_gpus(self):
        return self.num_nodes * self.world_size

    def train(self, dataloader: Union[DataLoader, DataModule] = None):
        import torch.multiprocessing as mp
        mp.spawn(mp_agent,
                 args=(self, super().train, dataloader),
                 nprocs=self.world_size)


class SingleMachineDDPDist(SingleMachineDDPSpawnDist):

    def train(self, dataloader: Union[DataLoader, DataModule]):
        if self.trainer.local_rank >= 0:
            mp_agent(self.trainer.local_rank, self.trainer, self.trainer.train, dataloader)
        else:
            self._call_children_scripts()

    def _call_children_scripts(self):
        # bookkeeping of spawned processes
        assert self.world_size == 0
        self._has_spawned_children = True

        # DDP Environment variables
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(find_free_network_port()))

        # allow the user to pass the node rank
        node_rank = "0"
        node_rank = os.environ.get("NODE_RANK", node_rank)
        node_rank = os.environ.get("GROUP_RANK", node_rank)
        os.environ["NODE_RANK"] = node_rank
        os.environ["LOCAL_RANK"] = "0"

        command = sys.argv
        command[0] = os.path.abspath(command[0])
        command = [sys.executable] + command

        os.environ["WORLD_SIZE"] = f"{self.world_size}"

        self.interactive_ddp_procs = []

        for local_rank in range(1, self.num_gpus):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            proc = subprocess.Popen(command, env=env_copy, cwd=None)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = random.randint(2, 5)
            time.sleep(delay)
