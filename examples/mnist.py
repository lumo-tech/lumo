import torch
from lumo import DatasetBuilder, MetricType, Trainer, TrainerParams, Meter, callbacks, DataModule
from lumo.proc.path import cache_dir
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, RandomCrop, Normalize
from torch import nn
from torch.nn import functional as F


def make_dataset():
    data = MNIST(root=cache_dir(), train=True, download=True)
    test_data = MNIST(root=cache_dir(), train=False, download=True)

    mean = torch.mean(data.data / 255.)
    std = torch.std(data.data / 255.)

    ds = (
        DatasetBuilder()
            .add_input('xs', data.data.float().unsqueeze(1))  # 注册样本来源，命名为 'xs'
            .add_input('ys', data.targets)  # 注册标签来源，命名为 'ys'
            .add_output('xs', 'xs0', transform=Normalize(mean=(mean,), std=(std,)))  # 添加一个弱增广输出 'xs0'
            .add_output('xs', 'xs1',
                        transform=Compose(
                            [RandomCrop(28, padding=4), Normalize(mean=(mean,), std=(std,))]))  # 添加一个强增广输出 'xs1'
            .add_output('ys', 'ys')  # 添加标签输出
    )
    print(ds)
    print(ds[0].keys())

    test_ds = (
        DatasetBuilder()
            .add_input('xs', test_data.data.float().unsqueeze(1))  # 注册样本来源，命名为 'xs'
            .add_input('ys', test_data.targets)  # 注册标签来源，命名为 'ys'
            .add_output('xs', 'xs', transform=Normalize(mean=(mean,), std=(std,)))  # 测试样本不使用增广
            .add_output('ys', 'ys')  # 添加标签输出
    )

    print(test_ds)
    print(test_ds[0].keys())
    return ds, test_ds


class MNISTParams(TrainerParams):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.optim = self.OPTIM.create_optim('SGD', lr=0.0001, momentum=0.9)


ParamsType = MNISTParams


class MNISTTrainer(Trainer):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.LoggerCallback().hook(self)

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        self.optim = params.optim.build(self.model.parameters())

        # manually trigger send_to_device method
        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        m = Meter()
        eval_xs, xs, ys = batch['xs0'], batch['xs1'], batch['ys']
        logits = self.model(xs)
        Lall = F.cross_entropy(logits, ys)
        self.optim.zero_grad()
        Lall.backward()
        self.optim.step()
        with torch.no_grad():
            m.mean.Lall = Lall
            eval_logits = self.model(eval_xs)
            m.mean.Ax = torch.eq(eval_logits.argmax(dim=-1), ys).float().mean()
        return m

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        m = Meter()
        xs, ys = batch['xs'], batch['ys']
        logits = self.model(xs)
        m.test_mean.Ax = torch.eq(logits.argmax(dim=-1), ys).float()
        return m


def main():
    ds, test_ds = make_dataset()

    # create datamodule to contain dataloader
    dl = ds.DataLoader(batch_size=32)
    test_dl = test_ds.DataLoader(batch_size=32)
    dm = DataModule()
    dm.regist_dataloader(train=dl, test=test_dl)

    params = MNISTParams()
    params.epoch = 10
    params.from_args()
    # with the input of params and dataloader, the initialization of models and optimizers in Trainer,
    # then the output will be the trained parameters, metrics and logs.
    trainer = MNISTTrainer(params, dm=dm)

    trainer.train()  # or trainer.train(dm=dl) if dm are not given above
    trainer.test()  # or trainer.test(dm=dl)
    trainer.save_last_model()


if __name__ == '__main__':
    main()
