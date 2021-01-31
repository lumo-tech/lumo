"""
Templete
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thexp import Trainer, Meter
from torch.nn import functional as F

from trainers import GlobalParams
from trainers.mixin import *


class BaseTrainer(callbacks.BaseCBMixin,
                  datasets.BaseSupDatasetMixin,
                  models.BaseModelMixin,
                  acc.ClassifyAccMixin,
                  losses.CELoss,
                  Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: GlobalParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        xs, ys = batch_data

        logits = self.to_logits(xs)

        meter.Lall = meter.Lall + self.loss_ce_(logits, ys, meter=meter, name='Lce')

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='acc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(torch.rand(xs.shape[0], params.n_classes))


if __name__ == '__main__':
    params = GlobalParams()
    # params.device = 'cuda:0'
    params.from_args()

    trainer = BaseTrainer(params)
    trainer.train()
    trainer.save_model()
