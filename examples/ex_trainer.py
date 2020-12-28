"""

"""
import sys

sys.path.insert(0, "../")
from thexp import __VERSION__

print(__VERSION__)

import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


from thexp.frame import Meter, Params, Trainer


class MyTrainer(Trainer):
    __exp_name__ = "DemoExp2"

    def callbacks(self, params: Params):
        from thexp import callbacks
        callbacks.LoggerCallback().hook(self)
        callbacks.AutoRecord().hook(self)

    def datasets(self, params: Params):
        from torchvision import transforms
        from torchvision.datasets import FakeData
        from thexp.contrib.data import DataLoader

        dataset = FakeData(size=32 * 10, image_size=(28, 28), transform=transforms.ToTensor())
        train_loader = eval_loader = test_loader = DataLoader(dataset, shuffle=True, batch_size=32, drop_last=True)

        self.regist_databundler(
            train=train_loader,
            test=test_loader,
            eval=eval_loader,
        )

    def models(self, params: Params):
        from torch.optim import SGD
        self.model = MyModel()
        self.optim = SGD(self.model.parameters(), lr=params.optim.lr)
        self.cross = nn.CrossEntropyLoss()

    def train_batch(self, eidx, idx, global_step, batch_data, params, device):
        optim, cross = self.optim, self.cross
        meter = Meter()
        xs, ys = batch_data

        # 训练逻辑
        logits = self.model(xs)
        meter.loss = cross(logits, ys)

        # 反向传播
        meter.loss.backward()
        optim.step()
        optim.zero_grad()

        return meter


params = Params()
params.epoch = 3

for p in params.grid_search("optim.lr", [0.1, 0.001, 0.0001]):
    trainer = MyTrainer(params)
    trainer.train()


MyTrainer.__exp_name__ = "DemoExp"
for p in params.grid_search("optim.lr", [0.1, 0.001, 0.0001]):
    trainer = MyTrainer(params)
    trainer.train()

# params.lr = 0.01

