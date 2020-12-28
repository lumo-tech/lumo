"""

"""
import torch
from torch import nn
from thexp import Params, Trainer


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class MyTrainer(Trainer):

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


def test_trainer():
    trainer = MyTrainer(Params())

    trainer.params.eidx = 3
    fn = trainer.save_keypoint()
    trainer.train()
    assert trainer.params.eidx == trainer.params.epoch
    trainer.load_checkpoint(fn)
    assert trainer.params.eidx == 3
