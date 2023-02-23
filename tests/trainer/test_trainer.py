from typing import Union, Optional, Sequence, Mapping, Any

import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader

from lumo import ParamsType, TrainerParams
from lumo import Trainer, DataModule, Meter, TrainStage, MetricType, Record, DatasetBuilder
from lumo.data.loader import DataLoaderType
from lumo.trainer import callbacks
from lumo.proc import glob


def create_dataset_builder():
    builder = (
        DatasetBuilder()
            .add_input(name='xs', source=range(1000))
            .add_input(name='ys', source=range(1, 1001))
            .add_output(name='xs', outkey='xs1')
            .add_output(name='xs', outkey='xs2')
            .add_output(name='ys', outkey='ys1')
            .add_output_transform('xs1', lambda x: x + 1)
            .add_output_transform('ys1', lambda x: x - 1)
    )
    return builder


class MyParams(TrainerParams):

    def __init__(self):
        super().__init__()
        self.epoch = 100


class LifecycleCallback(callbacks.TrainCallback):

    def __init__(self) -> None:
        super().__init__()
        self.c = 0
        self.call_order = []

    def on_train_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_train_begin')

    def on_train_epoch_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_train_epoch_begin')

    def on_test_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_test_begin')

    def on_eval_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_eval_begin')

    def on_train_step_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_train_step_begin')

    def on_eval_step_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_eval_step_begin')

    def on_test_step_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_test_step_begin')

    def on_train_end(self, trainer: Trainer, func, params: ParamsType, meter: Meter, *args, **kwargs):
        self.call_order.append('on_train_end')

    def on_train_epoch_end(self, trainer: Trainer, func, params: ParamsType, meter: Meter, *args, **kwargs):
        self.call_order.append('on_train_epoch_end')

    def on_predict_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_predict_begin')

    def on_test_end(self, trainer: Trainer, func, params: ParamsType, record: Record = None, *args, **kwargs):
        self.call_order.append('on_test_end')

    def on_eval_end(self, trainer: Trainer, func, params: ParamsType, record: Record = None, *args, **kwargs):
        self.call_order.append('on_eval_end')

    def on_train_step_end(self, trainer: Trainer, func, params: ParamsType, metric: MetricType = None, *args, **kwargs):
        self.call_order.append('on_train_step_end')

    def on_eval_step_end(self, trainer: Trainer, func, params: ParamsType, metric: MetricType = None, *args, **kwargs):
        self.call_order.append('on_eval_step_end')

    def on_test_step_end(self, trainer: Trainer, func, params: ParamsType, metric: MetricType = None, *args, **kwargs):
        self.call_order.append('on_test_step_end')

    def on_predict_end(self, trainer: Trainer, func, params: ParamsType, result: Any = None, *args, **kwargs):
        self.call_order.append('on_predict_end')

    def on_inference_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        self.call_order.append('on_inference_begin')

    def on_inference_end(self, trainer: Trainer, func, params: ParamsType, meter: Meter, *args, **kwargs):
        self.call_order.append('on_inference_end')


class CBTrainer(Trainer):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        lf = LifecycleCallback()
        lf.hook(self)
        self.lf = lf

        callbacks.LoggerCallback().hook(self)

    def stop_train(self):
        super().stop_train()

    def stop_train_epoch(self):
        super().stop_train_epoch()

    def to_device(self, item: Optional[Union[nn.Module, torch.Tensor, Sequence, Mapping]] = None,
                  device_args_kwargs=None):
        return super().to_device(item, device_args_kwargs)

    def prepare_dataloader(self, stage: TrainStage, dataloader=None):
        return super().prepare_dataloader(stage, dataloader)

    def initialize(self):
        super().initialize()

    def train(self, dm: Union[DataModule, DataLoaderType] = None, params: ParamsType = None, limit_global_steps=None):
        return super().train(dm, params, limit_global_steps)

    def train_epoch(self, loader: DataLoaderType, params: ParamsType = None, limit_step=None,
                    limit_global_steps=None) -> Record:
        return super().train_epoch(loader, params, limit_step, limit_global_steps)

    def remove_callback(self, cur):
        super().remove_callback(cur)

    def test(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step=None):
        return super().test(dm, params, limit_step)

    def evaluate(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step: int = None):
        return super().evaluate(dm, params, limit_step)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        super().test_step(batch, params)

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        super().evaluate_step(batch, params)


class MyDataModule(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)
        dl = create_dataset_builder().DataLoader()
        self.regist_dataloader_with_stage(stage, dl)


def test_callback():
    params = TrainerParams()

    glob['exp_root'] = tempfile.mkdtemp()
    glob['blob_root'] = tempfile.mkdtemp()
    glob['metric_root'] = tempfile.mkdtemp()
    glob['HOOK_LOCKFILE'] = False
    glob['HOOK_LASTCMD_DIR'] = tempfile.mkdtemp()
    glob['HOOK_GITCOMMIT'] = False
    glob['HOOK_RECORDABORT'] = False
    glob['HOOK_DIARY'] = False
    glob['HOOK_TIMEMONITOR'] = False
    glob['HOOK_FINALREPORT'] = False
    trainer = CBTrainer(params, dm=MyDataModule())
    trainer.train()
    trainer.logger.info(trainer.lf.call_order)
