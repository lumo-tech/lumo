"""
refer to
https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
"""
from typing import Union

import torch
from PIL import Image
from torch.utils.data import DataLoader

from lumo import DatasetBuilder, MetricType, Trainer, TrainerParams, Meter, callbacks, DataModule
from lumo.contrib import EMA, MemoryBank, StorageBank
from lumo.contrib.accelerate.utils import send_to_device
from lumo.contrib.nn.loss import contrastive_loss2
from lumo.proc.path import cache_dir
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms
from torchvision.models.resnet import resnet18
from torch import nn
from torch.nn import functional as F

"""define transforms"""


def none(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def simclr(mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


"""create datasets"""


def make_dataset():
    data = CIFAR10(root=cache_dir(), train=True, download=True)
    data.data = [Image.fromarray(img) for img in data.data]
    test_data = CIFAR10(root=cache_dir(), train=False, download=True)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    ds = (
        DatasetBuilder()
            .add_input('xs', data.data)  # 注册样本来源，命名为 'xs'
            .add_input('ys', data.targets)  # 注册标签来源，命名为 'ys'
            .add_output('xs', 'xs0', transform=simclr(mean, std))  # 添加一个弱增广输出 'xs0'
            .add_output('xs', 'xs1', transform=simclr(mean, std))  # 添加一个强增广输出 'xs1'
            .add_output('ys', 'ys')  # 添加标签输出
    )

    # for knn test
    memo_ds = (
        DatasetBuilder()
            .add_input('xs', data.data)  # 注册样本来源，命名为 'xs'
            .add_input('ys', data.targets)  # 注册标签来源，命名为 'ys'
            .add_output('xs', 'xs', transform=none(mean, std))  # 添加一个弱增广输出 'xs0'
            .add_output('ys', 'ys')  # 添加标签输出
    )
    print(ds)
    print(ds[0].keys())

    test_ds = (
        DatasetBuilder()
            .add_idx('idx')  # add index key for sample
            .add_input('xs', test_data.data)  # 注册样本来源，命名为 'xs'
            .add_input('ys', test_data.targets)  # 注册标签来源，命名为 'ys'
            .add_output('xs', 'xs', transform=none(mean, std))  # 测试样本不使用增广
            .add_output('ys', 'ys')  # 添加标签输出
    )

    print(test_ds)
    print(test_ds[0].keys())
    return ds, memo_ds, test_ds


class MocoParams(TrainerParams):
    def __init__(self):
        super().__init__()
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.06,
                                             momentum=0.9,
                                             weight_decay=5e-5,
                                             )
        self.lr_decay_end = 0.00001
        self.temperature = 0.1
        self.ema_alpha = 0.99
        self.feature_dim = 129
        self.queue_size = 4096
        self.batch_size = 512
        self.symmetric = False


ParamsType = MocoParams


class MocoModel(nn.Module):

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


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class MocoTrainer(Trainer):

    def icallbacks(self, params: ParamsType):
        callbacks.LoggerCallback().hook(self)

    def imodels(self, params: ParamsType):
        self.model = MocoModel(params.feature_dim)
        self.ema_model = EMA(self.model, alpha=params.ema_alpha)

        self.optim = params.optim.build(self.model.parameters())

        self.tensors = StorageBank()
        self.tensors.register('test_feature', dim=params.feature_dim, k=len(self.dm.test_dataset))
        self.tensors.register('test_ys', dim=-1, k=len(self.dm.test_dataset), dtype=torch.long)

        self.mem = MemoryBank()
        # do not need normalize because normalize is applied in contrastive_loss2 function
        self.mem.register('negative', dim=params.feature_dim, k=params.queue_size)
        self.mem['negative'] = F.normalize(self.mem['negative'], dim=-1)

        self.lr_sche = params.SCHE.Cos(
            start=params.optim.lr, end=params.lr_decay_end,
            left=0,
            right=len(self.train_dataloader) * params.epoch
        )
        # manually trigger send_to_device method
        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        m = Meter()
        im_query, im_key, ys = batch['xs0'], batch['xs1'], batch['ys']
        feat_query = self.model.forward(im_query)

        with torch.no_grad():
            # shuffle for making use of BN
            feat_key = self.ema_model.forward(im_key)  # keys: NxC
            feat_key = F.normalize(feat_key, dim=1)  # already normalized

        feat_query = F.normalize(feat_query, dim=1)

        if params.symmetric:
            Lcsa = contrastive_loss2(query=feat_query, key=feat_key,
                                     memory=self.mem['negative'],
                                     query_neg=False, key_neg=False,
                                     temperature=params.temperature,
                                     norm=False)
            Lcsb = contrastive_loss2(query=feat_key, key=feat_query,
                                     memory=self.mem['negative'],
                                     query_neg=False, key_neg=False,
                                     temperature=params.temperature,
                                     norm=False)
            Lcs = Lcsa + Lcsb
        else:

            Lcs = contrastive_loss2(query=feat_query, key=feat_key.detach(),
                                    memory=self.mem['negative'].clone().detach(),
                                    query_neg=False, key_neg=False,
                                    temperature=params.temperature,
                                    norm=False)

            # memory bank
        with torch.no_grad():
            if params.symmetric:
                self.mem.push('negative', torch.cat([feat_query, feat_key], dim=0))
            else:
                self.mem.push('negative', feat_key)

        self.optim.zero_grad()
        self.accelerate.backward(Lcs)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_steps)

        # metrics
        with torch.no_grad():
            m.mean.Lcs = Lcs
            m.last.lr = cur_lr
        return m

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        idx = batch['idx']
        xs, ys = batch['xs'], batch['ys']
        feature = self.model(xs)
        self.tensors.scatter('test_feature', feature, idx)
        self.tensors.scatter('test_ys', ys, idx)

    def test(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step=None):
        super().test(dm, params, limit_step)  # run default test loop
        self.save_last_model()

        feature_bank = []
        with torch.no_grad():
            # generate feature bank
            for batch in self.dm['memo']:
                batch = send_to_device(batch, self.device)
                data, target = batch['xs'], batch['ys']
                feature = self.model(data)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)

            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.tensor(self.dm['memo'].dataset.inputs['ys'], device=feature_bank.device)
            # loop test data to predict the label by weighted knn search

            pred_labels = knn_predict(self.tensors['test_feature'],
                                      feature_bank, feature_labels, params.n_classes, params.knn_k, params.knn_t)
            total_num = pred_labels.shape[0]
            total_top1 = torch.eq(pred_labels[:, 0], self.tensors['test_ys']).float().sum().item()

            knn_acc = total_top1 / total_num * 100

        max_knn_acc = self.metric.dump_metric('Knn', knn_acc, cmp='max', flush=True)
        self.logger.info(f'Best Knn Top-1 acc: {max_knn_acc}, current: {knn_acc}')

        if knn_acc >= max_knn_acc:
            self.save_best_model()


def main():
    ds, memo_ds, test_ds = make_dataset()

    params = MocoParams()
    params.from_args()

    # create datamodule to contain dataloader
    dl = ds.DataLoader(batch_size=params.batch_size)
    memo_dl = memo_ds.DataLoader(batch_size=params.batch_size)
    test_dl = test_ds.DataLoader(batch_size=params.batch_size)
    dm = DataModule()
    dm.regist_dataloader(train=dl,
                         test=test_dl,
                         memo=memo_dl)  # add extra dataloader with any name

    # with the input of params and dataloader, the initialization of models and optimizers in Trainer,
    # then the output will be the trained parameters, metrics and logs.
    trainer = MocoTrainer(params, dm=dm)

    trainer.train()  # or trainer.train(dm=dl) if dm are not given above
    trainer.test()  # or trainer.test(dm=dl)
    trainer.save_last_model()


if __name__ == '__main__':
    main()
