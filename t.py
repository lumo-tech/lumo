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

    def datasets(self, params: Params):
        from torchvision.datasets.fakedata import FakeData
        from torchvision.transforms import ToTensor
        dataset = FakeData(image_size=(3,32,32),transform=ToTensor())
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset,batch_size=32)
        self.regist_databundler(train=loader)

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
        raise NotImplemented()


def trainer():
    trainer = MyTrainer(Params())

    trainer.params.eidx = 3
    fn = trainer.save_keypoint()
    trainer.train()
    assert trainer.params.eidx == trainer.params.epoch+1
    trainer.load_checkpoint(fn)
    assert trainer.params.eidx == 3

if __name__ == '__main__':
    trainer()