from typing import Union, Optional, Sequence, Mapping

import torch
from torch import nn

from torch.utils.data import DataLoader

from lumo import Trainer, DataModule, callbacks, Params, Meter
from lumo.base_classes import TrainerStage
from lumo.kit import ParamsType, DataModuleMix


class TempCallback(callbacks.TrainCallback):

    def __init__(self) -> None:
        super().__init__()
        self.c = 0

    def on_train_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_train_epoch_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_test_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_eval_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_train_step_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_eval_step_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_test_step_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_train_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_test_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_eval_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_eval_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_test_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_predict_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_predict_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1

    def on_inference_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.c += 1

    def on_inference_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        self.c += 1


class CBTrainer(Trainer):
    def stop_train(self):
        super().stop_train()

    def stop_train_epoch(self):
        super().stop_train_epoch()

    def to_stage(self, stage: TrainerStage):
        super().to_stage(stage)

    def to_device(self, item: Optional[Union[nn.Module, torch.Tensor, Sequence, Mapping]] = None,
                  device_args_kwargs=None):
        return super().to_device(item, device_args_kwargs)

    def prepare_dataloader(self, stage: TrainerStage, dataloader=None):
        return super().prepare_dataloader(stage, dataloader)

    def train(self, dataloader: Union[DataLoader, DataModuleMix] = None):
        return super().train(dataloader)

    def train_epoch(self, dataloader: DataLoader):
        return super().train_epoch(dataloader)

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        super().train_step(idx, batch, params, *args, **kwargs)

    def evaluate(self, dataloader: Union[DataLoader, DataModule] = None):
        return super().evaluate(dataloader)

    def evaluate_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        super().evaluate_step(idx, batch, params, *args, **kwargs)

    def test(self, dataloader: Union[DataLoader, DataModule] = None):
        return super().test(dataloader)

    def test_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        super().test_step(idx, batch, params, *args, **kwargs)

    def inference(self, batch):
        super().inference(batch)

    def predict(self, batch):
        super().predict(batch)

    def ioptims(self, params: ParamsType):
        super().ioptims(params)

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)

    def imodels(self, params: ParamsType):
        super().imodels(params)


def test_callback():
    params = Params()

