import torch
from typing import List

from thexp import Trainer, Params, callbacks


class InnerTrainer(Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        res = self.model(batch_data)

        """
        处理是 list、iterater、单值的情况
        """

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


class Runner():
    def __init__(self, params: Params, train_dataloader=None, eval_dataloader=None, test_dataloader=None,
                 metric: List = None, callbacks: List[callbacks.BaseCallback] = None):
        self.trainer = InnerTrainer(params)
        self.metric = metric # TODO 参考 keras 的 Metric 如何绑定的
        self.callbacks = callbacks
        for cb in callbacks:
            cb.hook(self.trainer)

    def train(self, model, optim):
        self.trainer.model = model
        self.trainer.optim = optim

        self.trainer.train()


"""
"""
if __name__ == '__main__':
    train_dataloader = DataBundler(...)
    eval_dataloader = DataBundler(...)
    test_dataloader = DataBundler(...)

    model = ...

    params = ...

    optim = params.optim.build(...)

    trainer = Runner(params,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     test_dataloader=test_dataloader,
                     metric=[
                         ...  # functions
                     ],
                     callbacks=[
                         ...  # callbacks
                     ])
    trainer.train(model, optim)
