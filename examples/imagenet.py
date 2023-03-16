import sys
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import DataLoader
import os

from torchvision.datasets.folder import default_loader

from lumo import DatasetBuilder, MetricType, Trainer, TrainerParams, Meter, callbacks, DataModule
from torchvision.datasets import FakeData, ImageFolder
from torchvision import transforms
from torchvision.models.resnet import resnet18
from torch import nn
from lumo.proc.dist import is_dist, is_main
from torch.nn import functional as F
from lumo.proc import glob
from lumo.utils.subprocess import run_command

"""define transforms"""


def none(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def standard(mean, std, resize=None):
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


"""create datasets"""


def imagenet(split='train'):
    """
    download from https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description
    ```
    mkdir imagenet
    cd ./imagenet
    kaggle competitions download -c imagenet-object-localization-challenge
    unzip imagenet-object-localization-challenge.zip
    tar -xvf imagenet_object_localization_patched2019.tar.gz
    ls
    >>> ILSVRC LOC_synset_mapping.txt  LOC_val_solution.csv imagenet_object_localization_patched2019.tar.gz
    >>> LOC_sample_submission.csv  LOC_train_solution.csv  imagenet-object-localization-challenge.zip
    ```
    """
    root = glob['imagenet']
    if split == 'train':
        file = Path(root).joinpath('ILSVRC', 'ImageSets', 'CLS-LOC', 'train_cls.txt')
        train_root = os.path.join(root, 'ILSVRC/Data/CLS-LOC/train')
        with file.open('r') as r:
            lines = r.readlines()
            imgs = [line.split(' ')[0] for line in lines]
            name_cls_map = {name: i for i, name in enumerate(sorted(set([i.split('/')[0] for i in imgs])))}
            xs = [os.path.join(train_root, f'{i}.JPEG') for i in imgs]
            ys = [name_cls_map[i.split('/')[0]] for i in imgs]
    else:
        file = Path(root).joinpath('LOC_val_solution.csv')
        val_root = os.path.join(root, 'ILSVRC/Data/CLS-LOC/val')

        with file.open('r') as r:
            r.readline()
            lines = r.readlines()
            lines = [line.split(',') for line in lines]
            lines = [[img, res.split(' ')[0]] for img, res in lines]

            name_cls_map = {name: i for i, name in enumerate(sorted(set([i[1] for i in lines])))}
            xs = [os.path.join(val_root, f'{img}.JPEG') for img, _ in lines]
            ys = [name_cls_map[res] for _, res in lines]

    return list(xs), list(ys)


def take_first(item):
    return item[0]


def take_second(item):
    return item[1]


def make_dataset(dummy=False):
    if dummy:
        train_dataset = FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
        ds = (
            DatasetBuilder()
                .add_input('fake', train_dataset)
                .add_output('fake', 'xs', transform=take_first)
                .add_output('fake', 'ys', transform=take_second)
        )
        test_ds = (
            DatasetBuilder()
                .add_input('fake', val_dataset)
                .add_output('fake', 'xs', transform=take_first)
                .add_output('fake', 'ys', transform=take_second)
        )
    else:
        train_dataset = ImageFolder(os.path.join(glob['imagenet'], 'train'))
        val_dataset = ImageFolder(os.path.join(glob['imagenet'], 'val'))

        xs, ys = list(zip(*train_dataset.samples))
        test_xs, test_ys = list(zip(*val_dataset.samples))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        ds = (
            DatasetBuilder()
                .add_input('xs', xs, transform=default_loader)  # 注册样本来源，命名为 'xs'
                .add_input('ys', ys)  # 注册标签来源，命名为 'ys'
                .add_output('xs', 'xs', transform=standard(mean, std))  # 添加一个弱增广输出 'xs0'
                .add_output('ys', 'ys')  # 添加标签输出
        )

        print(ds)
        print(ds[0].keys())

        test_ds = (
            DatasetBuilder()
                .add_input('xs', test_xs, transform=default_loader)  # 注册样本来源，命名为 'xs'
                .add_input('ys', test_ys)  # 注册标签来源，命名为 'ys'
                .add_output('xs', 'xs', transform=none(mean, std))  # 测试样本不使用增广
                .add_output('ys', 'ys')  # 添加标签输出
        )

    print(test_ds)
    print(test_ds[0].keys())
    return ds, test_ds


class LargeParams(TrainerParams):
    def __init__(self):
        super().__init__()
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.06,
                                             momentum=0.9,
                                             weight_decay=5e-5,
                                             )
        self.lr_decay_end = 0.00001
        self.batch_size = 512
        self.dummy = False
        self.multiprocessing_distributed = True


ParamsType = LargeParams


class LargeModel(nn.Module):

    def __init__(self, feature_dim) -> None:
        super().__init__()
        self.backbone = resnet18()
        in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_feature, feature_dim, bias=True)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        feature = self.head(feature_map)
        return feature


class LargeTrainer(Trainer):

    def icallbacks(self, params: ParamsType):
        callbacks.LoggerCallback().hook(self)

    def imodels(self, params: ParamsType):
        self.model = resnet18(num_classes=1000)
        self.optim = params.optim.build(self.model.parameters())

        self.lr_sche = params.SCHE.Cos(
            start=params.optim.lr, end=params.lr_decay_end,
            left=0,
            right=len(self.train_dataloader) * params.epoch
        )
        # manually trigger send_to_device method
        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        m = Meter()
        xs, ys = batch['xs'], batch['ys']
        logits = self.model(xs)

        Lall = F.cross_entropy(logits, ys)
        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()

        # change lr by training epoch
        cur_lr = self.lr_sche.apply(self.optim, self.eidx)

        with torch.no_grad():
            m.mean.Lall = Lall
            m.mean.Ax = torch.eq(logits.argmax(dim=-1), ys).float().mean()
            m.last.lr = cur_lr
        return m

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        m = Meter()
        xs, ys = batch['xs'], batch['ys']
        logits = self.model(xs)

        all_logits = self.accelerate.gather(logits)
        all_ys = self.accelerate.gather(ys)

        m.test_mean.Ax = torch.eq(all_logits.argmax(dim=-1), all_ys).float()
        return m


def main():
    params = LargeParams()
    params.from_args()

    if params.multiprocessing_distributed and not is_dist():
        command = ' '.join([
            'accelerate', 'launch',
            *sys.argv,
        ])
        run_command(command)
    else:  # not distributed or in distribution environment
        # create datamodule to contain dataloader
        ds, test_ds = make_dataset(dummy=params.dummy)
        dl = ds.DataLoader(batch_size=params.batch_size, num_workers=4)
        test_dl = test_ds.DataLoader(batch_size=params.batch_size, num_workers=4)
        dm = DataModule()
        dm.regist_dataloader(train=dl, test=test_dl)

        # with the input of params and dataloader, the initialization of models and optimizers in Trainer,
        # then the output will be the trained parameters, metrics and logs.
        trainer = LargeTrainer(params, dm=dm)

        trainer.train()  # or trainer.train(dm=dl) if dm are not given above
        trainer.test()  # or trainer.test(dm=dl)
        trainer.save_last_model()


if __name__ == '__main__':
    main()
