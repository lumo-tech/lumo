"""

"""
import subprocess
import time
import sys
import bisect
import inspect
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, TypeVar, Union, Optional

# from ..utils.lazy import torch, np
import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lumo.base_classes import attr
from lumo.base_classes.metaclasses import Merge
from lumo.kit.datamodule import DataModule
from lumo.kit.environ import globs
from lumo.kit.experiment import TrainerExperiment
from lumo.kit.logger import Logger
from lumo.kit.meter import Meter, AvgMeter
from lumo.kit.params import Params, DistributionParams
from lumo.utils.keys import TRAINER
from lumo.utils.connect import find_free_network_port

ParamsType = TypeVar('ParamsType', bound=Params)

from enum import Enum


@dataclass()
class initial_tuple():
    models: bool = False
    callbacks: bool = False
    optims: bool = False


class TrainerStage(Enum):
    init = -1
    train = 0
    test = 1
    evaluate = 2


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


class _LoopImp():
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

    def train_step(self, idx, batch):
        raise NotImplementedError()

    def evaluate(self, dataloader: Union[DataLoader, DataModule]):
        raise NotImplementedError()

    def evaluate_step(self, idx, batch) -> Union[Dict, Meter]:
        raise NotImplementedError()

    def test(self, dataloader: Union[DataLoader, DataModule]):
        raise NotImplementedError()

    def test_step(self, idx, batch) -> Union[Dict, Meter]:
        raise NotImplementedError()

    def inference(self, batch):
        raise NotImplementedError()

    def predict(self, batch):
        """alias of inference"""
        raise NotImplementedError()


class _BaseTrainer(metaclass=Merge):
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
        "train", "train_epoch", "train_step", "test", "eval", "train_on_batch", "train_batch", "predict",
        "load_keypoint", "load_checkpoint", "load_model", "save_keypoint", "save_checkpoint", "save_model",
    }

    _check_initial = {
        'train', 'text', 'eval'
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
                        # TODO 添加 rich 输出，更优雅的退出
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
        self.initial = initial_tuple()
        self.train_epoch_toggle = False
        self.train_toggle = False

        if 'device' in params:
            _device = torch.device(params.device)
            self.regist_device(_device)

        self._check_cb_init()
        self._initialize_globs()

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
    def device(self) -> torch.device:
        return self._state_dicts[TRAINER.DKEY.device]

    @property
    def models(self) -> Dict[str, nn.Module]:
        self._check_models_init()
        return self._state_dicts[TRAINER.DKEY.models]

    @property
    def optims(self) -> Dict[str, Optimizer]:
        self._check_optim_init()
        return self._state_dicts[TRAINER.DKEY.optims]

    @property
    def buffer(self) -> Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]:
        return self._state_dicts[TRAINER.DKEY.tensor]

    @property
    def nparr(self) -> Dict[str, np.ndarray]:
        return self._state_dicts[TRAINER.DKEY.tensor]['np']

    @property
    def tharr(self) -> Dict[str, torch.Tensor]:
        return self._state_dicts[TRAINER.DKEY.tensor]['th']

    @property
    def others(self) -> Dict[str,]:
        return self._state_dicts[TRAINER.DKEY.others]

    def regist_device(self, device: torch.device):
        self._state_dicts[TRAINER.DKEY.device] = device
        if device.type == 'cuda':
            torch.cuda.set_device(device)

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

    def change_mode(self, train=True):
        for k, v in self.models.items():
            if train:
                v.train()
            else:
                v.eval()

    def optim_state_dict(self):
        return {k: v.state_dict() for k, v in self.optims.items()}

    def model_state_dict(self):
        return {k: v.state_dict() for k, v in self.models.items()}

    def buffer_state_dict(self):
        return self.buffer

    def other_state_dict(self):
        return {k: v.state_dict() for k, v in self.others.items()}

    def inner_state_dict(self):
        return {
            self._state_dicts
        }

    def state_dict(self):
        res = {
            'models': self.model_state_dict(),
            'optims': self.optim_state_dict(),
            'buffer': self.buffer_state_dict(),
            'other': self.other_state_dict(),
        }
        for k, v in self._state_dicts.items():
            if k not in res:
                res[k] = v

        return res

    def _load_fun_state_dict(self, src: dict, tgt: dict):
        for k, v in tgt:
            if k in src:
                v.load_state_dict(src[k])

    def load_state_dict(self, state_dict: dict):
        _sub = {'models', 'optims', 'other'}
        for k, v in state_dict.items():
            if k in _sub:
                self._load_fun_state_dict(v, self._state_dicts[k])
            else:
                self._state_dicts[k] = v

    def ioptims(self, params: ParamsType):
        pass

    def icallbacks(self, params: ParamsType):
        """初始化回调函数"""
        pass

    def imodels(self, params: ParamsType):
        """初始化模型"""
        pass


class Trainer(_LoopImp, _BaseTrainer):

    def _check_dist_environ(self, loader: DataLoader):
        if self.is_dist:
            from torch.nn.parallel import DistributedDataParallel
            from torch.utils.data.distributed import DistributedSampler
            from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
            for k in self.models:
                model = self.models[k]
                if not isinstance(model, DistributedDataParallel):
                    model = DistributedDataParallel(model,
                                                    find_unused_parameters=True,
                                                    device_ids=[self.local_rank])
                self.models[k] = model

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
            setattr(loader, f'_{loader.__class__.__name__}__initialized', True)

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

    def train(self, dataloader: Union[DataLoader, DataModule]):
        self.to_stage(TrainerStage.train)
        if isinstance(dataloader, DataModule):
            dataloader = dataloader.train_dataloader
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
        return TrainerResult(TrainerStage.train, 0)

    def train_epoch(self, dataloader: DataLoader):
        avg = AvgMeter()
        for idx, batch in enumerate(dataloader):
            meter = self.train_step(idx, batch)
            if isinstance(meter, (dict, Meter)):
                avg.update(meter)
            self.params.global_step += 1
            self.params.idx = idx
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

    def train_step(self, idx, batch):
        pass

    def evaluate(self, dataloader: Union[DataLoader, DataModule]):
        self.to_stage(TrainerStage.evaluate)
        if isinstance(dataloader, DataModule):
            dataloader = dataloader.val_dataloader
        if dataloader is None:
            return TrainerResult(TrainerStage.evaluate, 1, 'no eval_dataloader')

        self._check_dist_environ(dataloader)
        avg = AvgMeter()
        for idx, batch in enumerate(dataloader):
            meter = self.evaluate_step(idx, batch)
            avg.update(meter)
        return TrainerResult(TrainerStage.evaluate, 0)

    def evaluate_step(self, idx, batch) -> Union[Dict, Meter]:
        return self.test_step(idx, batch)

    def test(self, dataloader: Union[DataLoader, DataModule]):
        self.to_stage(TrainerStage.test)
        if isinstance(dataloader, DataModule):
            dataloader = dataloader.test_dataloader
        if dataloader is None:
            return TrainerResult(TrainerStage.test, 1, 'no test_dataloader')

        self._check_dist_environ(dataloader)
        avg = AvgMeter()
        for idx, batch in enumerate(dataloader):
            meter = self.test_step(idx, batch)
            avg.update(meter)
        return TrainerResult(TrainerStage.test, 0)

    def test_step(self, idx, batch) -> Union[Dict, Meter]:
        pass

    def inference(self, batch):
        pass

    def predict(self, batch):
        """alias of inference"""
        self.inference(batch)


class SingleMachineDDPSpawnDist(_LoopImp):
    pcls = DistributionParams

    def __init__(self, trainer: Trainer, params: DistributionParams):
        self.trainer = trainer
        self.params = params
        if params.world_size == -1:
            params.world_size = torch.cuda.device_count()

    @property
    def num_nodes(self):
        return 1

    @property
    def world_size(self):
        return self.num_nodes * self.num_gpus

    @property
    def num_gpus(self):
        return self.num_nodes * self.world_size

    def train(self, dataloader: Union[DataLoader, DataModule]):
        import torch.multiprocessing as mp
        mp.spawn(mp_agent,
                 args=(self.trainer, self.trainer.train, dataloader),
                 nprocs=self.world_size)


class SingleMachineDDPDist(SingleMachineDDPSpawnDist):

    def train(self, dataloader: Union[DataLoader, DataModule]):
        self._call_children_scripts()

    def _call_children_scripts(self):
        # bookkeeping of spawned processes
        assert self.global_rank == 0
        self._check_can_spawn_children()
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
            delay = np.random.uniform(1, 5, 1)[0]
            time.sleep(delay)
