import bisect
import warnings
from functools import lru_cache
from typing import Union, Dict, Any, Optional, Sequence, Mapping

import numpy as np
import torch
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import send_to_device
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lumo.data.accelerator import DataLoaderShard, DataLoaderDispatcher
from lumo.core import ParamsType, TrainStage, Record, MetricType, Meter, Attr
from lumo.data import DataModule
from lumo.proc import dist

from lumo.trainer.rnd import RndManager
from lumo.utils.logger import Logger
from accelerate import Accelerator
from .base import _BaseTrainer
from .components import TrainerExperiment
from .saver import Saver
from ..core.table import TableRow
from ..data.loader import DataLoaderType, DataLoaderSide


class Trainer(_BaseTrainer):
    callback_function = {
        "save_keypoint", "save_checkpoint", "save_model", "load_state_dict",
        'ioptims', 'imodels', 'train', 'test', 'evaluate', 'train_epoch', 'train_step',
        'test_step', 'evaluate_step', 'process_loader', 'regist_dataloader'
    }

    def __init__(self, params: ParamsType, dm: DataModule = None):
        if dm is None:
            dm = DataModule(params)
        else:
            if dm.params is None:
                dm.params = params

        self.dm = dm  # type: DataModule
        self.params = params
        self._logger = None
        self._saver = None
        self.shared_prop = {}
        self.params.iparams()
        self.exp = TrainerExperiment(self.generate_exp_name())

        self._database = TableRow(self.exp.exp_name, self.exp.test_name_with_dist)

        self.rnd = RndManager()

        self.train_epoch_toggle = False
        self.train_toggle = False

        device = params.get('device', None) if not self.is_dist else None

        self.accelerate = Accelerator(kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=params.get('find_unused_parameters', False))
        ])

        if self.accelerate.state.distributed_type == self.accelerate.state.distributed_type.NO:
            self.accelerate.state.device = torch.device(device)

        if dist.is_main():
            self.params.to_yaml(self.exp.test_file('params.yaml'))

        self.set_global_steps(params.get('global_steps', 0))
        if params.get('debug', False):
            self.exp.set_prop('debug', True)

    def regist_dataloader(self, dataloader: DataLoader, stage: TrainStage):
        self.datamodule.regist_dataloader_with_stage(stage, dataloader)

    def process_loader(self, dm: Union[DataModule, DataLoader] = None, stage: TrainStage = TrainStage.train):
        """
        automatically called before train()/test()/evaluate(), see __new__ function of Trainer
        :param dm:
        :param stage:
        :return:
        """
        assert stage is not None, '`stage` cannot be None'
        if dm is None and self.dm is not None:
            dm = self.dm

        if isinstance(dm, DataModule):
            loader = dm[stage.value]
            if loader is None:
                # where datamodule.idataloader() methods first invoked (automaticly).
                loader = dm.get_loader_with_stage(stage)
                if loader is None:
                    return None
                loader = self.prepare_dataloader(loader, stage)
                self.regist_dataloader(loader, stage=stage)
        elif isinstance(dm, DataLoader):
            loader = dm
            loader = self.prepare_dataloader(loader, stage)
            self.regist_dataloader(loader, stage=stage)
        else:
            return None

        return loader

    def _load_fun_state_dict(self, src: dict, tgt: dict):
        for k, v in tgt.items():
            if k in src:
                v.load_state_dict(src[k])

    def load_state_dict(self, state_dict: dict):
        _sub = {'models', 'optims', 'other'}
        _missing = []

        for k, v in state_dict.items():
            if k in _sub:
                self._load_fun_state_dict(v, self._state_dicts[k])
            else:
                self._state_dicts[k] = v
        return

    @property
    def db(self):
        return self._database

    @property
    def database(self):
        return self._database

    @property
    def saver(self) -> Saver:
        if self._saver is None:
            self._saver = Saver(self.exp.saver_dir)
        return self._saver

    @property
    def is_debug(self):
        return self.params.get('debug', False)

    @property
    def is_main(self):
        return dist.is_main()

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
    def wandb(self):
        import wandb
        wandb.init(project=self.exp.exp_name, name=self.exp.test_name)
        wandb.config = self.params.to_dict()
        return wandb

    @property
    def logger(self):
        if self._logger is None:
            from lumo.utils.logger import set_global_logger
            self._logger = Logger()
            set_global_logger(self._logger)
            if self.params.get('debug', False):
                self._logger.set_verbose(Logger.V_DEBUG)
                self._logger.debug('Enable debug log.')
            if self.is_main:
                fn = self._logger.add_log_dir(self.exp.log_dir)
                self.exp.dump_info('logger_args', {'log_dir': fn})
        return self._logger

    @property
    def safe_writer(self):
        """see trainer.writer"""
        try:
            import tensorflow as tf
            import tensorboard as tb
            tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        except ImportError:
            pass

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
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            from tensorboardX import SummaryWriter

        kwargs = self.exp.board_args
        res = SummaryWriter(**kwargs)

        def close(*args):
            res.flush()
            res.close()

        self.exp.add_exit_hook(close)
        return res

    @property
    def first_epoch(self):
        return self.eidx == 0

    @property
    def first_step(self):
        return self.global_steps == 0

    @property
    def idx(self):
        return self._prop.get('idx', 0)

    @property
    def eidx(self):
        return self._prop.get('eidx', 0)

    @property
    def trainer_state(self):
        return self._prop

    @property
    def devices(self) -> Dict[str, torch.device]:
        return self._state_dicts['devices']

    @property
    def model_dict(self) -> Dict[str, nn.Module]:
        return {key: self[key]
                for key in self._state_dicts['models']}

    @property
    def optim_dict(self) -> Dict[str, Optimizer]:

        return {key: self[key] for key in self._state_dicts['optims']}

    @property
    def torch_tensor(self) -> Dict[str, torch.Tensor]:
        return {key: self[key] for key in self._state_dicts['tensor.th']}

    @property
    def numpy_tensor(self) -> Dict[str, np.ndarray]:
        return {key: self[key] for key in self._state_dicts['tensor.np']}

    @property
    def others(self) -> Dict[str, Any]:
        return {key: self[key] for key in self._state_dicts['others']}

    @property
    def datamodule(self) -> DataModule:
        return self.dm

    @property
    def train_dataloader(self) -> Optional[DataLoaderType]:
        return self.datamodule['train']

    @property
    def test_dataloader(self) -> Optional[DataLoaderType]:
        return self.datamodule['test']

    @property
    def val_dataloader(self) -> Optional[DataLoaderType]:
        return self.datamodule['val']

    @property
    def global_steps(self) -> int:
        return self._prop['global_steps']

    @property
    def device(self):
        return self.accelerate.device

    # def prepare(self):
    #     pass

    def to_device(self, item: Optional[Union[nn.Module, torch.Tensor, Sequence, Mapping]] = None,
                  device: torch.device = None):

        if item is None:
            for k, v in list(self.model_dict.items()):
                self[k] = self.accelerate.prepare(v)
            for k, v in list(self.optim_dict.items()):
                self[k] = self.accelerate.prepare(v)
        else:
            if device is None:
                device = self.device
            item = send_to_device(item, device)
            return item

    def initialize(self):
        if self._prop.get('initial', False):
            return
        self.exp.start()
        self.icallbacks(self.params)
        self.set_property('initial.callbacks', True)
        self.imodels(self.params)
        self.set_property('initial.model', True)
        self.set_property('initial', True)

    def stop_train(self):
        self.train_toggle = True
        self.train_epoch_toggle = True

    def stop_train_epoch(self):
        self.train_epoch_toggle = True

    def prepare_dataloader(self, loader: DataLoaderType, stage: TrainStage = None):
        """
        automatically called before train()/test()/evaluate(), see __new__ function of Trainer
        :param loader:
        :param stage:
        :return:
        """
        if isinstance(loader, (DataLoaderShard, DataLoaderDispatcher)):
            warnings.warn('Duplicated prepare a same DataLoader twice, check your code.')
            return loader

        split_batches = self.params.get('split_batches', None)
        if stage is not None and not stage.is_train():
            split_batches = True

        """do not change original loader stage"""
        if isinstance(loader, DataLoader):
            self.accelerate.split_batches = split_batches
            loader = self.accelerate.prepare_data_loader(loader)
        elif isinstance(loader, DataLoaderSide):
            loader = loader.copy()
            loader._loaders = {k: self.prepare_dataloader(v, stage) for k, v in loader._loaders.items()}
        return loader

    def train(self, dm: Union[DataModule, DataLoaderType] = None, params: ParamsType = None, limit_global_steps=None):
        loader = self.select_loader(dm)
        if not loader:
            loader = self.train_dataloader

        if loader is None:
            self.set_property('early_stop', 'Lack of train loader')
            return self._prop

        if params is None:
            params = self.params

        for eidx in range(params.epoch):
            self.set_epoch_idx(eidx)
            epoch_record = self.train_epoch(loader, params, limit_global_steps=limit_global_steps)
            self.set_property('record', epoch_record)
            self.set_property('record', epoch_record)
            if self.train_toggle:
                self.set_property('early_stop', 'train toggle')
                self.train_toggle = False
                break
            if limit_global_steps is not None and self.global_steps >= limit_global_steps:
                self.set_property('early_stop', f'meet limit_global_steps {limit_global_steps}')
                break
        return self._prop

    def train_epoch(self, loader: DataLoaderType, params: ParamsType = None,
                    limit_step=None,
                    limit_global_steps=None) -> Record:
        stage = TrainStage.train
        self.change_stage(stage)
        record = self.create_record(stage=stage)

        if params is None:
            params = self.params

        self.wait_for_everyone()
        for idx, batch in enumerate(loader):
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

            if limit_step is not None and idx > limit_step:
                break
            if limit_global_steps is not None and self.global_steps >= limit_global_steps:
                break

            self.set_idx(idx)
            self._prop['global_steps'] += 1
            metric = self.train_step(batch, params)
            record.record(metric)

        for k, v in record.agg().items():
            self.share(f'train_epoch.{k}', v)

        record.flush()
        return record

    def set_property(self, key, value):
        self._prop[key] = value

    def set_global_steps(self, val):
        self.set_property('global_steps', val)

    def set_epoch_idx(self, val):
        self.set_property('eidx', val)

    def set_idx(self, val):
        self.set_property('idx', val)

    @property
    def trainstage(self) -> TrainStage:
        return self._prop.get('stage', TrainStage.default)

    def set_stage(self, val: TrainStage):
        self.set_property('stage', val)

    def add_callback(self, callback):
        """
        添加一个回调函数，注意，不能添加重复的 callback，这不推荐，也没有必要。
        :param callback:
        :return:
        """
        msg = None
        cb_name = callback.__class__.__name__
        cb_names = {cb.__class__.__name__ for cb in self.callbacks}
        if callback not in self.callbacks and cb_name in cb_names:
            msg = "Callback duplicate."
            callback.on_hook_failed(self, msg)
            return False

        if callback.only_main_process and not self.is_main:
            msg = f"{callback.__class__.__name__} only_main_process but in local_rank {self.local_rank}"
            callback.on_hook_failed(self, msg)
            return False

        if callback.only_single_gpu and not self.is_dist:
            msg = f"{callback.__class__.__name__} only_single_gpu but dist={self.is_dist}"
            callback.on_hook_failed(self, msg)
            return False

        if msg is not None:
            return False
        bisect.insort(self.callbacks, callback)

        callback._hooked = self
        callback.on_hooked(self, self.params)
        self.logger.info(f'{callback} hooked')
        return True

    def remove_callback(self, cur):
        self.callbacks.remove(cur)
        pass

    def change_stage(self, stage: TrainStage):
        if self.trainstage == stage:
            return

        self.set_stage(stage)
        for k, v in self.model_dict.items():
            if 'ema' in k.lower():
                continue
            if stage.value:
                v.train()
            else:
                v.eval()

    @classmethod
    def select_loader(cls, dm=None):
        loader = None
        if dm:
            if isinstance(dm, DataModule):
                loader = dm.train_dataloader
            elif isinstance(dm, DataLoader) or isinstance(dm, DataLoaderSide):
                loader = dm
            else:
                raise TypeError(type(dm))
        return loader

    def test(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step=None):
        stage = TrainStage.test
        self.change_stage(stage)

        loader = self.select_loader(dm)
        if not loader:
            loader = self.test_dataloader

        if loader is None:
            return None

        if params is None:
            params = self.params

        record = self.create_record(stage=stage)
        self.wait_for_everyone()
        for idx, batch in enumerate(loader):
            if limit_step is not None and idx >= limit_step:
                break
            self.set_idx(idx)
            metric = self.test_step(batch, params)
            record.record(metric)
        for k, v in record.agg().items():
            self.share(f'test.{k}', v)
        record.flush()
        return record

    def evaluate(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step: int = None):
        stage = TrainStage.val
        self.change_stage(stage)

        loader = self.select_loader(dm)
        if not loader:
            loader = self.val_dataloader
        if loader is None:
            return None

        if params is None:
            params = self.params

        record = self.create_record(stage=stage)
        for idx, batch in enumerate(loader):
            if limit_step is not None and idx >= limit_step:
                break
            self.set_idx(idx)
            metric = self.evaluate_step(batch, params)
            record.record(metric)
        for k, v in record.agg().items():
            self.share(f'test.{k}', v)
        record.flush()
        return record

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        pass

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        pass

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        pass

    def imodels(self, params: ParamsType):
        pass

    def icallbacks(self, params: ParamsType):
        pass

    def inference(self, batch):
        raise NotImplementedError()

    def predict(self, batch):
        raise NotImplementedError()

    def optim_state_dict(self, wrap=True):
        res = {k: v.state_dict() for k, v in self.optim_dict.items()}
        if wrap:
            res = {'optim': res}
        return res

    def model_state_dict(self, wrap=True):
        res = {k: self.accelerate.unwrap_model(v).state_dict() for k, v in self.model_dict.items()}
        if wrap:
            res = {'model': res}
        return res

    def other_state_dict(self, wrap=True):
        res = {k: v.state_dict() for k, v in self.others.items()}
        if wrap:
            res = {'other': res}
        return res

    def state_dict(self):
        res = {
            'optims': self.optim_state_dict(wrap=False),
            'models': self.model_state_dict(wrap=False),
            'others': self.other_state_dict(wrap=False),
            'thtensor': self.torch_tensor,
            'nptensor': self.numpy_tensor,
        }

        return res

    def Meter(self):
        return Meter()

    def create_record(self, stage: TrainStage = None):
        if stage is None:
            stage = self.trainstage
        record = Record(stage=stage)
        return record

    def wait_for_everyone(self):
        """
        making sure all processes have reached this point before continuing.
        """
        self.accelerate.wait_for_everyone()

    def save_model(self, is_best=False, meta_info: Union[str, dict, Attr] = None):
        info = self._build_trainer_meta_info(meta_info)
        val = self.saver.save_model(self.eidx, self.model_state_dict(),
                                    meta_info=info,
                                    is_best=is_best)
        self.wait_for_everyone()
        return val

    def _build_trainer_meta_info(self, meta_info: Union[str, dict, Attr] = None):
        info = dict()
        info['eidx'] = self.eidx
        if meta_info is not None:
            if isinstance(meta_info, str):
                info['msg'] = meta_info
            if isinstance(meta_info, Meter):
                meta_info = meta_info.serialize()
            if isinstance(meta_info, dict):
                info.update(meta_info)
        return info

    def save_checkpoint(self, max_keep=10, is_best=False, meta_info: Union[str, dict, Meter] = None):
        info = self._build_trainer_meta_info(meta_info)
        val = self.saver.save_checkpoint(self.eidx, self.state_dict(),
                                         meta_info=info,
                                         max_keep=max_keep,
                                         is_best=is_best)
        self.wait_for_everyone()
        return val

    def share(self, key, value):
        self.shared_prop[key] = value
