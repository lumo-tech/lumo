"""
一个训练 f(x) = x+1 的线性函数的例子
"""
from lumo import Trainer, DatasetBuilder, Params, callbacks
from lumo.kit.params import ParamsType

import torch
from torch import nn

from lumo import Meter


class PlusOneTrainer(Trainer):

    def imodels(self, params: ParamsType):
        self.model = nn.Linear(1, 1)
        self.optim = params.optim.build(self.model.parameters())
        self.to_device()

    def icallbacks(self, params: ParamsType):
        callbacks.LoggerCallback().hook(self)

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        meter = Meter()
        xs, ys = batch
        logits = self.model(xs.float())
        loss = torch.mean((ys - logits) ** 2)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        meter.Lce = loss
        return meter


params = Params()
params.epoch = 30
params.batch_size = 10
params.optim = params.OPTIM.create_optim('SGD', lr=0.000001, weight_decay=4e-3)

trainer = PlusOneTrainer(params)

builder = (
    DatasetBuilder().add_input('xs', range(-500, 500)).add_input('ys', range(-500, 500))
        .add_output('xs', 'xs').add_output('ys', 'ys')
        .add_output_transform('xs', lambda x: torch.tensor([x]))
        .add_output_transform('ys', lambda x: torch.tensor([x + 1]))
        .random_sampler().chain()
)

builder = (
    DatasetBuilder(
        xs=range(-500, 500),
        ys=range(-500, 500),
        auto_output=True,
        xs_opt_transform=lambda x: torch.tensor([x]),
        ys_opt_transform=lambda x: torch.tensor([x + 1])
    ).random_sampler().chain()
)  # another simpler way

trainer.train(builder.DataLoader(batch_size=params.batch_size, num_workers=4))

print(list(trainer.model.parameters()))
