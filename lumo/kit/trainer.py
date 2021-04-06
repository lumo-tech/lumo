"""

"""
import bisect
import inspect
import os
from collections.abc import Iterator
from functools import wraps
from typing import Any, Dict, TypeVar

# from ..utils.lazy import torch, np
import numpy as np
import torch
from torch import distributed as dist
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import nn
from lumo.base_classes import attr
from lumo.base_classes.metaclasses import Merge
from lumo.kit.logger import Logger
from lumo.kit.saver import Saver
from lumo.kit.builder import Dataset, DatasetWrap, BaseBuilder
from lumo.kit.environ import globs
from lumo.kit.experiment import TrainerExperiment
from lumo.kit.meter import Meter, AvgMeter
from lumo.kit.params import Params, BaseParams
from lumo.utils.keys import TRAINER

ParamsType = TypeVar('ParamsType', bound=Params)


def _gene_class_exp_name(trainer_instance) -> str:
    try:
        file = inspect.getfile(trainer_instance.__class__)
        pre = os.path.splitext(os.path.basename(file))[0]
    except:
        pre = 'builtin'

    return "{}.{}".format(pre, trainer_instance.__exp_name__)


def _gene_trainer_info(trainer_instance) -> dict:
    try:
        path = inspect.getfile(trainer_instance.__class__)
        basefn = os.path.splitext(os.path.basename(path))[0]
    except:
        path = 'built_in'
        basefn = 'built_in'
    from lumo.utils.keys import TRAINER
    trainer_kwargs = {
        TRAINER.path: path,
        TRAINER.doc: trainer_instance.__class__.__doc__,
        TRAINER.basename: basefn,
        TRAINER.class_name: trainer_instance.__class__.__name__
    }

    return trainer_kwargs


def mp_agent(rank, trainer, op):
    import torch.distributed as dist
    trainer.params.local_rank = rank
    dist.init_process_group(backend='nccl', init_method=trainer.params.init_method,
                            rank=rank,
                            world_size=trainer.params.world_size)

    trainer.params.device = 'cuda:{}'.format(rank)
    trainer.regist_device(torch.device(trainer.params.device))
    torch.cuda.set_device(trainer.params.local_rank)
    print('in rank {}'.format(rank))
    trainer.imodels(trainer.params)
    trainer.idatasets(trainer.params)
    trainer.icallbacks(trainer.params)
    op(trainer)


class BaseTrainer(metaclass=Merge):
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
        "train", "train_epoch", "train_step", "test", "eval", "train_on_batch",
        "regist_databundler", "train_batch", "test_eval_logic", "test_eval_logic_v2", "predict",
        "load_keypoint", "load_checkpoint", "load_model",
        "save_keypoint", "save_checkpoint", "save_model",
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

    def __init__(self, params: ParamsType):
        self._state_dicts = attr({
            "models": {},
            "optims": {},
            "buffers": {},
            "others": {},
            'device': torch.device('cpu'),
            'devices': {},
            'initialized': {
                'models': False,
                'datasets': False,
                'dataloader': False,
            },
            'train_epoch_toggle': False,
            'train_toggle': False,
            'params': params,
            'exp': TrainerExperiment(_gene_class_exp_name(self)),
        })
        self._saver = None
        self._logger = None

        self._datasets = attr()  # type:attr[str,BaseBuilder]
        self._dataloaders = attr()  # type:attr[str,DataLoader]

        # self.experiment = TrainerExperiment(_gene_class_exp_name(self))  # type: TrainerExperiment

        if 'device' in params:
            _device = torch.device(params.device)
            self.regist_device(_device)

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
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            self._state_dicts[TRAINER.DKEY.buffers][name] = value
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            self._state_dicts[TRAINER.DKEY.others][name] = value

    def __setitem__(self, key: str, value: Any):
        self.__setattr__(key, value)

    def __setstate__(self, state):
        self._state_dicts = attr.from_dict(state)

    def __getstate__(self):
        res = self._state_dicts.pickify()
        return res

    def __getattr__(self, item):
        if item in self._state_dicts:
            return self._state_dicts[item]
        return self.__getattribute__(item)

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except:
            return getattr(self._state_dicts, name)

    def _initialize_globs(self):
        if dist.is_available() and dist.is_initialized():
            globs['rank'] = dist.get_rank()
            globs['world_size'] = dist.get_world_size()
        else:
            globs['rank'] = -1
            globs['world_size'] = 0

        for k, v in self.params.items():  # type:str,Any
            if k.isupper():
                globs[k] = v
                if isinstance(v, str):
                    os.environ[k] = v

    def _initialize_dataloader(self):
        for k, v in self.datasets.items():  # type: str,BaseBuilder
            if k in self._dataloaders:
                continue

            dataloader = v.build_dataloader()  # TODO
            self._dataloaders[k] = dataloader

        self._state_dicts['initialized']['dataloader'] = True

    @property
    def train_toggle(self):
        return self._state_dicts['train_toggle']

    @property
    def train_epoch_toggle(self):
        return self._state_dicts['train_epoch_toggle']

    @property
    def exp(self) -> TrainerExperiment:
        return self._state_dicts[TRAINER.DKEY.exp]

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = Logger()
            self._logger.add_log_dir(self.exp.log_dir)
        return self._logger

    @property
    def params(self) -> ParamsType:
        return self._state_dicts[TRAINER.DKEY.params]

    @property
    def saver(self) -> Saver:
        if self._saver is None:
            self._saver = Saver(self.exp.saver_dir)
        return self._saver

    @property
    def visdom(self):
        import visdom
        vis = visdom.Visom(env=self.exp.test_name)
        return vis

    @property
    def safe_writer(self):
        """see trainer.writer"""
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        return self.writer

    @property
    def writer(self):
        from torch.utils.tensorboard import SummaryWriter

        kwargs = self.exp.board_args
        res = SummaryWriter(**kwargs)

        def close(*args, **kwargs):
            res.flush()
            res.close()

        self.exp.add_exit_hook(close)
        return res

    @property
    def device(self) -> torch.device:
        return self._state_dicts[TRAINER.DKEY.device]

    @property
    def datasets(self) -> attr[str, BaseBuilder]:
        if not self.isinit_datasets:
            self.idatasets(params=self.params)
        return self._datasets

    @property
    def dataloaders(self):
        if not self.isinit_dataloader:
            self._initialize_dataloader()
        return self._dataloaders

    @property
    def train_dataloader(self):
        return self.dataloaders.get('train', None)

    @property
    def val_dataloader(self):
        return self.dataloaders.get('val', None)

    @property
    def test_dataloader(self):
        return self.dataloaders.get('test', None)

    @property
    def models(self) -> Dict[str, nn.Module]:
        if not self.isinit_models:
            self.imodels(params=self.params)
        return self._state_dicts['models']

    @property
    def optims(self) -> Dict[str, Optimizer]:
        if not self.isinit_models:
            self.imodels(params=self.params)
        return self._state_dicts['optims']

    @property
    def buffers(self) -> attr:
        return self._state_dicts['buffers']

    @property
    def others(self) -> attr:
        return self._state_dicts['others']

    @property
    def devices(self) -> attr[str, torch.device]:
        return self._state_dicts['devices']

    @property
    def isinit_dataloader(self) -> bool:
        return self._state_dicts['initialized']['dataloader']

    @property
    def isinit_datasets(self) -> bool:
        return self._state_dicts['initialized']['datasets']

    @property
    def isinit_models(self) -> bool:
        return self._state_dicts['initialized']['models']

    # initialize and construct methods

    def initialize(self):
        if not self.isinit_models:
            self.imodels(self.params)
            return
        if not self.isinit_datasets:
            self.idatasets(self.params)
        self.icallbacks(self.params)
        self._state_dicts['initialized'] = True

    def icallbacks(self, params: Params):
        """初始化回调函数"""
        pass

    def idatasets(self, params: Params):
        """初始化数据集"""
        pass

    def imodels(self, params: Params):
        """初始化模型"""
        pass

    def ioptim(self,params:Params):
        """"""
        pass

    def regist_device(self, device: torch.device):
        self._state_dicts[TRAINER.DKEY.device] = device
        if device.type == 'cuda':
            torch.cuda.set_device(device)

    def regist_dataset(self, train: BaseBuilder = None, val: BaseBuilder = None, test: BaseBuilder = None,
                       **others: BaseBuilder):
        if train is not None:
            self._datasets['train'] = DatasetWrap.check_then_wrap(train)
        if val is not None:
            self._datasets['val'] = DatasetWrap.check_then_wrap(val)
        if test is not None:
            self._datasets['test'] = DatasetWrap.check_then_wrap(test)

        for k, v in others.items():
            self._datasets[k] = DatasetWrap.check_then_wrap(v)

    def regist_dataloader(self, train: DataLoader = None, val: DataLoader = None,
                          test: DataLoader = None,
                          **others: DataLoader):
        if train is not None:
            self._dataloaders['train'] = train
        if val is not None:
            self._dataloaders['val'] = val
        if test is not None:
            self._dataloaders['test'] = test

        for k, v in others.items():
            self._dataloaders[k] = v

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
        return True

    def reload_callback(self, callback):
        self.remove_callback(callback.__class__)
        return self.add_callback(callback)

    def remove_callback(self, callback):
        """"""
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

    # train/evalutaion/test logical
    def toggle_train(self, flag: bool = None):
        if flag is None:
            flag = not self.train_toggle
        self._state_dicts['train_toggle'] = flag

    def toggle_train_epoch(self, flag: bool = None):
        if flag is None:
            flag = not self.train_epoch_toggle
        self._state_dicts['train_epoch_toggle'] = flag

    def train(self):
        params = self.params
        while params.eidx < params.epoch:
            self.train_epoch(params.eidx, params=self.params)
            params.eidx += 1
            if self.train_toggle:
                self.toggle_train(False)
                break

    def train_epoch(self, eidx, params: ParamsType):
        avg = AvgMeter()
        self.change_mode(True)
        for idx, batch_data in enumerate(self.train_dataloader):
            params.idx = idx
            res = self.train_batch(eidx, idx, params.global_step, batch_data, params)
            if res is not None:
                avg.update(res)
            params.global_step += 1
            if self.train_epoch_toggle:
                self.toggle_train_epoch(False)
                break

    def test(self):
        loader = self.test_dataloader
        if loader is None:
            return

        avg = AvgMeter()
        for idx, batch_data in enumerate(loader):
            res = self.inference(batch_data)
            avg.update(res)
        return avg

    def evaluate(self):
        loader = self.val_dataloader
        if loader is None:
            return

        avg = AvgMeter()
        for idx, batch_data in enumerate(loader):
            res = self.inference(batch_data)
            avg.update(res)
        return avg

    def predict(self, batch):
        """alias of inference"""
        self.inference(batch)

    def train_batch(self, eidx, idx, global_steps, batch_data, params: ParamsType):
        raise NotImplementedError()

    def inference(self, batch):
        raise NotImplementedError()

    def inference_sample(self, sample):
        raise NotImplementedError()

    # state preservation

    def load_checkpoints(self, fn=None, obj: dict = None):
        if fn is not None:
            obj = self.saver.load_state_dict(fn)
        if obj is None:
            raise ValueError(f'{fn} is invalid file.')
        for k in self._state_dicts:
            if k not in {'models', 'optims',
                         'device', 'devices',
                         'initialized',
                         'train_epoch_toggle', 'train_toggle',
                         'exp'}:
                if k in obj:
                    self._state_dicts[k] = obj[k]

        def _load_state_dict(src, tgt):
            for k, v in src.items():
                if k in tgt:
                    tgt[k].load_state_dict(v)

        if 'models' in obj:
            models = self.models
            _load_state_dict(obj['models'], models)
        if 'optims' in obj:
            optims = self.optims
            _load_state_dict(obj['optims'], optims)

    def save_checkpoints(self, meta_info: dict = None):
        res = {}
        for k, v in self._state_dicts.items():
            if k not in {'models', 'optims',
                         'device', 'devices',
                         'initialized',
                         'train_epoch_toggle', 'train_toggle',
                         'exp'}:
                res[k] = v
        res['models'] = self.models_state_dict()
        res['optims'] = self.optims_state_dict()

    def load_models_state_dict(self, fn=None, obj: dist = None):
        if fn is not None:
            obj = self.saver.load_state_dict(fn)
        if obj is None:
            raise ValueError(f'{fn} is invalid file.')

        for k, v in self.models.items():
            if k in obj:
                v.load_state_dict(obj[k])
                obj.pop(k)

    def optims_state_dict(self):
        res = {}
        for k, v in self.optims.items():
            res[k] = v.state_dict()
        return res

    def models_state_dict(self):
        res = {}
        for k, v in self.models.items():
            res[k] = v.state_dict()
        return res

    def save_models_state_dict(self, meta_info: dict = None):
        models_state_dict = self.models_state_dict()
        fn = self.saver.save_model(self.params.eidx, models_state_dict,
                                   meta_info=meta_info)
        return fn


class Trainer(BaseTrainer):

    def icallbacks(self, params: Params):
        pass

    def idatasets(self, params: Params):
        pass

    def imodels(self, params: Params):
        pass

    def train_batch(self, eidx, idx, global_steps, batch_data, params: ParamsType):
        raise NotImplementedError()

    def inference(self, batch):
        raise NotImplementedError()


class WrapTrainer(Trainer):

    def __init__(self, params: Params, train_dataloader, model_with_loss_fn, optimize, eval_dataloader=None,
                 test_dataloader=None):
        super().__init__(params)


class DistributedTrainer():
    def __init__(self, trainer_cls, params: Params, op):
        self.trainer_cls = trainer_cls
        self.params = params
        self.op = op

    def _ini_dist_models(self, trainer: BaseTrainer):
        # trainer.models
        pass

    def run(self):
        trainer = self.trainer_cls(self.params)  # type:Trainer
        # trainer = self.trainer_cls.__new__(self.trainer_cls)   # type:Trainer

        params = self.params
        params.initial()

        params.distributed = True
        import torch.multiprocessing as mp
        if params.world_size == -1:
            params.world_size = torch.cuda.device_count()

        mp.spawn(mp_agent,
                 args=(trainer, self.op),
                 nprocs=trainer.params.world_size)
