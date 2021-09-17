"""
一个训练 f(x) = x+1 的线性函数的例子
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lumo import Trainer, DatasetBuilder, Params, callbacks, DataModule
from lumo.base_classes import TrainerStage
from lumo.kit.params import ParamsType

import torch
from torch import nn

from lumo import Meter


class DM(DataModule):

    def idataloader(self, params: ParamsType, stage: TrainerStage, repeat: bool = False):
        super().idataloader(params, stage, repeat)
        print('repeat', repeat)
        loader = builder.DataLoader(batch_size=params.batch_size, num_workers=4)
        self.regist_dataloader_with_stage(stage, loader)

    def iidataloader(self, params: ParamsType, stage: TrainerStage, repeat: bool = False):
        super().iidataloader(params, stage, repeat)
        print(repeat)


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


#
# params = Params()
# params.epoch = 30
# params.batch_size = 10
# params.optim = params.OPTIM.create_optim('SGD', lr=0.000001, weight_decay=4e-3)
#
# trainer = PlusOneTrainer(params)
#
# builder = (
#     DatasetBuilder().add_input('xs', range(-500, 500)).add_input('ys', range(-500, 500))
#         .add_output('xs', 'xs').add_output('ys', 'ys')
#         .add_output_transform('xs', lambda x: torch.tensor([x]))
#         .add_output_transform('ys', lambda x: torch.tensor([x + 1]))
#         .random_sampler().chain()
# )
#
# builder = (
#     DatasetBuilder(
#         xs=range(-500, 500),
#         ys=range(-500, 500),
#         auto_output=True,
#         xs_opt_transform=lambda x: torch.tensor([x]),
#         ys_opt_transform=lambda x: torch.tensor([x + 1])
#     ).random_sampler().chain()
# )  # another simpler way
#
# params.epoch = 10
# trainer.train()
# trainer.train(DM())
#
# params.eidx = 0
# trainer.train()
# print(list(trainer.model.parameters()))


# from lumo.calculate.schedule import *
# from lumo import Logger
# import numpy as np
#
# learning_rate = 2e-4
# epoches_stage1 = 5
# epoches_stage2 = 5
# epoches_stage3 = 20
# train_dataloader = [200] * 200
# lr_scheduler = SchedulerList(
#     [
#         CosScheduler(start=learning_rate, end=1e-6, left=0, right=epoches_stage1),
#         CosScheduler(start=learning_rate, end=1e-6, left=epoches_stage1,
#                      right=(epoches_stage1 + epoches_stage2)),
#         CosScheduler(start=learning_rate, end=1e-6, left=(epoches_stage1 + epoches_stage2),
#                      right=(epoches_stage1 + epoches_stage2 + epoches_stage3))
#     ]
# )
#
# # # epoch = 44
# # idx = np.linspace(-1, 40, num=20000)
# # from matplotlib import pyplot as plt
# #
# # print()
# # print(lr_scheduler(0))
# # plt.plot([lr_scheduler(i) for i in idx])
# # plt.show()
# #
# # log = Logger()
# # from lumo import Meter, AvgMeter
# #
# # avg = AvgMeter()
# #
# # for i in idx:
# #     # print(i,lr_scheduler(i))
# #     avg.lr = lr_scheduler(i)
# #     log.inline(avg)
# #
# # cos = CosScheduler(start=learning_rate, end=1e-6, left=2,
# #                    right=2)
# # c = ConstantScheduler(4)
# # print(c(0))
# #
# # print(cos(1))
# #
# # cos.plot()
#
# from transformers.models.bart import BartForConditionalGeneration, BartConfig, BartTokenizer
#
# config = BartConfig(encoder_layers=2, decoder_layers=3, d_model=768)
#
# model = BartForConditionalGeneration(config)
#
# import torch
#
# x = torch.rand(3, 32, 768)
# #
# # cls_token_ids = torch.ones_like(x[:, 0:1], dtype=torch.long) * BartTokenizer().cls_token_id
# #
# # cls_emb = model.get_input_embeddings()(cls_token_ids)
# # torch.cat([cls_emb, x], dim=1)
#
#
# outputs = model.forward(
#     inputs_embeds=x,
#     output_hidden_states=True, labels=torch.randint(0, 32, (3, 32)))
#
# print(outputs.encoder_hidden_states[-1])
# print(outputs.encoder_hidden_states[-3])
# print((outputs.encoder_last_hidden_state == outputs.encoder_hidden_states[-1]).all())
# print(outputs)
from typing import Any, Optional, Union
from typing import List

from rich.console import Console, JustifyMethod, OverflowMethod, NewLine
from rich.segment import Segment
from rich.style import Style

from rich.progress import track
import time

from rich.progress import (Progress, TimeRemainingColumn, TextColumn, TimeElapsedColumn, Column, StyleType,
                           SpinnerColumn)

# with Progress() as progress:
#
#     task1 = progress.add_task("[red]Downloading...", total=1000)
#     task2 = progress.add_task("[green]Processing...", total=1000)
#     task3 = progress.add_task("[cyan]Cooking...", total=1000)
#
# while not progress.finished:
#     progress.update(task1, advance=0.5)
#     progress.update(task2, advance=0.3)
#     progress.update(task3, advance=0.9)
#     time.sleep(0.02)
from lumo.proc.date import strftime

columns = []

columns.extend(
    (
        TextColumn("{task.fields[time]}"),
        TimeElapsedColumn(),
        "|",
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "|",
        TextColumn("{task.fields[meter]}"),
        "|",
        TimeRemainingColumn(),
    )
)
progress = Progress(
    *columns,
)

from lumo import AvgMeter

avg = AvgMeter()
epoch_task = progress.add_task('idx', meter=avg, time=strftime)

with progress:
    while not progress.finished:
        avg.lr = random.random()
        avg.a = random.random()
        progress.update(epoch_task, advance=1)
        time.sleep(0.2)
    # yield from progress.track(
    #     sequence, total=total, description=description, update_period=update_period
    # )

# for i in console.render('4600/4635 | Lall: 0.4351 | lr: 0.0004525906'):
#     print(i)

# for i in range(10):
#     print('\r12313', end='')
