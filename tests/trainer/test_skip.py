from typing import Union, Optional, Sequence, Mapping, Any

import tempfile

import torch

from lumo.proc.config import debug_mode
from lumo.proc.path import cache_dir
from torch import nn
from torch.utils.data import DataLoader

from lumo import ParamsType, TrainerParams
from lumo import Trainer, DataModule, Meter, TrainStage, MetricType, Record, DatasetBuilder
from lumo.data.loader import DataLoaderType
from lumo import callbacks
from lumo.proc import glob


def create_dataset_builder():
    builder = (
        DatasetBuilder()
            .add_input(name='xs', source=range(1000))
            .add_input(name='ys', source=range(1, 1001))
            .add_output(name='xs', outkey='xs1')
            .add_output(name='xs', outkey='xs2')
            .add_output(name='ys', outkey='ys1')
            .set_output_transform('xs1', lambda x: x + 1)
            .set_output_transform('ys1', lambda x: x - 1)
    )
    return builder


class MyParams(TrainerParams):

    def __init__(self):
        super().__init__()
        self.epoch = 2


class CBTrainer(Trainer):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.SkipWhenParamsEq().hook(self)

    def imodels(self, params: ParamsType):
        super().imodels(params)
        assert self.context == 'imodels'

    def stop_train(self):
        super().stop_train()

    def stop_train_epoch(self):
        super().stop_train_epoch()

    def to_device(self, item: Optional[Union[nn.Module, torch.Tensor, Sequence, Mapping]] = None,
                  device_args_kwargs=None):
        return super().to_device(item, device_args_kwargs)

    def prepare_dataloader(self, loader: DataLoaderType, stage: TrainStage = None):
        return super().prepare_dataloader(loader, stage)

    def initialize(self):
        if not self.is_initialized:
            assert self.eidx == 0
            assert self.global_steps == 0
            assert self.idx == 0
            assert self.is_main
            assert self.first_step
            assert self.first_epoch
        super().initialize()
        assert self.is_initialized

    def train(self, dm: Union[DataModule, DataLoaderType] = None, params: ParamsType = None, limit_global_steps=None):
        assert self.context == 'train'
        return super().train(dm, params, limit_global_steps)

    def train_epoch(self, loader: DataLoaderType, params: ParamsType = None, limit_step=None,
                    limit_global_steps=None) -> Record:
        assert self.context == 'train_epoch'
        assert self.contexts[-2] == 'train'
        return super().train_epoch(loader, params, limit_step, limit_global_steps)

    def remove_callback(self, cur):
        super().remove_callback(cur)
        assert self.context == 'remove_callback'

    def test(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step=None):
        assert self.context == 'test'
        return super().test(dm, params, limit_step)

    def evaluate(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step: int = None):
        assert self.context == 'evaluate'
        return super().evaluate(dm, params, limit_step)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        assert self.context == 'train_step'
        assert self.contexts[-2] == 'train_epoch'
        assert self.contexts[-3] == 'train'
        super().train_step(batch, params)

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        assert self.context == 'test_step'
        assert self.contexts[-2] == 'test'
        super().test_step(batch, params)

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        assert self.context == 'evaluate_step'
        assert self.contexts[-2] == 'evaluate'
        super().evaluate_step(batch, params)


class MyDataModule(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)
        dl = create_dataset_builder().DataLoader()
        self.regist_dataloader_with_stage(stage, dl)


def test_callback():
    params = MyParams()
    params.epoch = 2

    debug_mode()
    trainer = CBTrainer(params, dm=MyDataModule())
    trainer.train()
    trainer.test()
    trainer.evaluate()
    trainer.exp.end()

    assert trainer.eidx == 1

    nparams = MyParams()
    assert nparams.hash() == params.hash()

    trainer = CBTrainer(nparams, dm=MyDataModule())
    trainer.train()
    assert trainer.eidx == 0

    n1params = MyParams()
    n1params.epoch = 3
    assert (n1params.hash() != nparams.hash())
    trainer = CBTrainer(n1params, dm=MyDataModule())
    trainer.train()
    assert trainer.eidx == 2
