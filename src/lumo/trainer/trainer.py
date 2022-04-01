import bisect
from functools import lru_cache
from typing import Union, Dict, Any, Optional, Sequence, Mapping

import numpy as np
import torch
from accelerate.utils import send_to_device
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lumo.core import ParamsType, TrainStage, Record, MetricType, Meter, Attr
from lumo.data import DataModule
from lumo.proc import dist
from lumo.proc.dist import is_main, is_dist
from lumo.trainer.rnd import RndManager
from lumo.utils.logger import Logger
from .accelerator import Accelerator
from .base import _BaseTrainer
from .components import TrainerExperiment
from .saver import Saver


class Trainer(_BaseTrainer):
    callback_function = {
        "save_keypoint", "save_checkpoint", "save_model", "load_state_dict",
        'ioptims', 'imodels', 'train', 'test', 'val', 'train_epoch', 'train_step',
        'test_step', 'val_step', 'process_loader', 'regist_dataloader'
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
        self.params.iparams()
        self.exp = TrainerExperiment(self._gene_class_exp_name())
        self.rnd = RndManager()

        self.train_epoch_toggle = False
        self.train_toggle = False

        self.accelerate = Accelerator(device=params.get('device', None))
        self.set_global_steps(params.get('global_steps', 0))
        if params.get('debug', False):
            self.exp.set_prop('debug', True)

    def regist_dataloader(self, dataloader: DataLoader, stage: TrainStage):
        self.datamodule.regist_dataloader_with_stage(stage, dataloader)

    def process_loader(self, dm: Union[DataModule, DataLoader] = None, stage: TrainStage = TrainStage.train):
        assert stage is not None, '`stage` cannot be None'
        if dm is None and self.dm is not None:
            dm = self.dm

        if isinstance(dm, DataModule):
            loader = dm[stage.value]
            if loader is None:
                loader = dm.get_loader_with_stage(stage)
                if loader is None:
                    return None
                self.regist_dataloader(loader, stage=stage)
        elif isinstance(dm, DataLoader):
            loader = dm
            self.regist_dataloader(loader, stage=stage)
        else:
            return None

        loader = self.accelerate.prepare_data_loader(loader)
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
    def saver(self) -> Saver:
        if self._saver is None:
            self._saver = Saver(self.exp.saver_dir)
        return self._saver

    @property
    def is_debug(self):
        return self.params.get('debug', False)

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
    def logger(self):
        if self._logger is None:
            from lumo.utils.logger import set_global_logger
            self._logger = Logger()
            set_global_logger(self._logger)
            if self.params.get('debug', False):
                self._logger.set_verbose(Logger.V_DEBUG)
                self._logger.debug('Enable debug log.')
            if is_main():
                self._logger.add_log_dir(self.exp.log_dir)
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
        from torch.utils.tensorboard import SummaryWriter

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
        return self._state_dicts['models']

    @property
    def optim_dict(self) -> Dict[str, Optimizer]:
        return self._state_dicts['optims']

    @property
    def torch_tensor(self) -> Dict[str, torch.Tensor]:
        return self._state_dicts['tensor.th']

    @property
    def numpy_tensor(self) -> Dict[str, np.ndarray]:
        return self._state_dicts['tensor.np']

    @property
    def others(self) -> Dict[str, Any]:
        return self._state_dicts['others']

    @property
    def datamodule(self) -> DataModule:
        return self.dm

    @property
    def train_dataloader(self):
        return self.dm['train']

    @property
    def test_dataloader(self):
        return self.dm['test']

    @property
    def val_dataloader(self):
        return self.dm['val']

    @property
    def global_steps(self):
        return self._prop['global_steps']

    @property
    def record_backend(self):
        return self.params.get('record', {}).get('dbrecord', 500)

    @property
    def record_window_size(self):
        return self.params.get('record', {}).get('window_size', 500)

    @property
    def device(self):
        return self.accelerate.device

    def to_device(self, item: Optional[Union[nn.Module, torch.Tensor, Sequence, Mapping]] = None,
                  device: torch.device = None):
        if item is None:
            for k, v in list(self.model_dict.items()):
                self.model_dict[k] = self.accelerate.prepare(v)
            for k, v in list(self.optim_dict.items()):
                self.optim_dict[k] = self.accelerate.prepare(v)
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

    def train(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_global_steps=None):
        loader = self.train_dataloader

        if loader is None:
            self.set_property('early_stop', 'Lack of train loader')
            return self._prop

        if params is None:
            params = self.params
        for eidx in range(params.epoch):
            self.set_epoch_ids(eidx)
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

    def train_epoch(self, loader: DataLoader, params: ParamsType = None,
                    limit_step=None,
                    limit_global_steps=None) -> Record:
        stage = TrainStage.train
        self.change_stage(stage)
        record = self.create_record(stage=stage)

        for idx, batch in enumerate(loader):
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

            if limit_step is not None and idx > limit_step:
                break
            if limit_global_steps is not None and self.global_steps >= limit_global_steps:
                break
            metric = self.train_step(batch, params)
            self._prop['global_steps'] += 1
            self._prop['idx'] = idx
            record.record(metric)
        record.flush()
        return record

    def set_property(self, key, value):
        self._prop[key] = value

    def set_global_steps(self, val):
        self.set_property('global_steps', val)

    def set_epoch_ids(self, val):
        self.set_property('eidx', val)

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

        if callback.only_main_process and not is_main():
            msg = f"{callback.__class__.__name__} only_main_process but in local_rank {self.local_rank}"
            callback.on_hook_failed(self, msg)
            return False

        if callback.only_single_gpu and not is_dist():
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

    def test(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step=None):
        stage = TrainStage.test
        self.change_stage(stage)

        loader = self.test_dataloader
        if loader is None:
            return None

        record = self.create_record(stage=stage)
        for idx, batch in enumerate(loader):
            self._prop['idx'] += idx
            if limit_step is not None and idx >= limit_step:
                break
            metric = self.test_step(batch, params)
            record.record(metric)
        record.flush()
        return record

    def evaluate(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step: int = None):
        stage = TrainStage.val
        self.change_stage(stage)
        loader = self.val_dataloader
        if loader is None:
            return None

        record = self.create_record(stage=stage)
        for idx, batch in enumerate(loader):
            self._prop['idx'] += idx
            if limit_step is not None and idx >= limit_step:
                break
            metric = self.evaluate_step(batch, params)
            record.record(metric)
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
            'optims': self.optim_state_dict(),
            'models': self.model_state_dict(),
            'others': self.other_state_dict(),
            'thtensor': self.torch_tensor,
            'nptensor': self.numpy_tensor,
        }

        return res

    def Meter(self):
        return Meter()

    def create_record(self, stage: TrainStage = None, prefix=None, blob=False, **kwargs):
        if stage is None:
            stage = self.trainstage

        if prefix is None:
            prefix = ''
        else:
            prefix = f'{prefix}.'

        if stage is None:
            stage = ''
        else:
            stage = f'{stage.value}.'

        if blob:
            fn = self.exp.blob_file(f'{stage}{prefix}{self.global_steps:06d}.rd', 'metric')
        else:
            fn = self.exp.test_file(f'{stage}{prefix}{self.global_steps:06d}.rd', 'metric')

        kwargs.setdefault('backend', self.record_backend)
        kwargs.setdefault('window_size', self.record_window_size)

        record = Record(location=fn,
                        **kwargs)
        return record

    def wait_for_everyone(self):
        """
        making sure all processes have reached this point before continuing.
        """
        self.accelerate.wait_for_everyone()

    def save_model(self, is_best=False, meta_info: Union[str, dict, Attr] = None):
        info = self._build_trainer_meta_info(meta_info)
        val = self.saver.save_model(self.eidx, {'models': self.model_state_dict()},
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
