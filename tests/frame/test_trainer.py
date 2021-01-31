"""

"""
import torch
from torch import nn
from thexp import Params, Trainer, callbacks, Meter


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class TCallbacks(callbacks.TrainCallback):
    def __init__(self):
        self.mark = set()

    def on_initial_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_initial_end(trainer, func, params, meter, *args, **kwargs)
        self.mark.add(1)


class MyTrainer(Trainer):

    def callbacks(self, params: Params):
        super().callbacks(params)
        TCallbacks().hook(self)

    def models(self, params: Params):
        super().models(params)
        self.rnd.mark('test')
        self.modela = MyModel()
        self.rnd.mark('test')
        self.modelb = MyModel()
        for pa, pb in zip(self.modela.parameters(), self.modelb.parameters()):
            assert (pa.data == pb.data).all()

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)


def get_params():
    p = Params()
    p.git_commit = False
    return p


def test_trainer():
    trainer = MyTrainer(get_params())

    trainer.params.eidx = 3
    fn = trainer.save_keypoint()

    trainer.params.eidx = 0
    trainer.load_checkpoint(fn)
    assert trainer.params.eidx == 3


def test_callbacks():
    trainer = MyTrainer(get_params())

    assert 1 in list(trainer._callback_set)[0].mark
