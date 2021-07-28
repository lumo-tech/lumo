"""
"""
from torchvision.models.resnet import resnet34
from torch.nn import functional as F
from torch import nn
from lumo import Trainer, Params, DatasetBuilder, DataModule, callbacks, Meter
from lumo.base_classes import TrainerStage
from lumo.proc import path


class DemoPM(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 20
        self.batch_size = 128
        self.num_workers = 8
        self.pin_memory = True

        self.pretrain = True
        self.pretrain_path = None
        self.dataset = self.choice('cifar10', 'cifar100')
        self.optim = self.OPTIM.create_optim('SGD', weight_decay=0.001, lr=0.01, momentum=0.9)
        self.dataset_root = path.dataset_cache_dir('demo')
        self.download_dataset = True
        self.lr_sche = None
        self.n_classes = 10

    def iparams(self):
        super().iparams()
        if self.dataset == 'cifar100':
            self.n_classes = 100

        self.lr_sche = self.SCHE.Cos(start=0, end=self.epoch,
                                     left=self.optim.args.lr,
                                     right=1e-6)


ParamsType = DemoPM


class CVDM(DataModule):

    def idataloader(self, params: ParamsType, stage: TrainerStage, repeat: bool = False):
        from torchvision.datasets import cifar
        if params.dataset == 'cifar10':
            dataset = cifar.CIFAR10(params.dataset_root, train=stage.is_train, download=params.download_dataset)
        else:
            dataset = cifar.CIFAR100(params.dataset_root, train=stage.is_train, download=params.download_dataset)

        ds = (
            DatasetBuilder()
                .add_input('xs', dataset.data)
                .add_input('ys', dataset.targets)
                .add_output('xs', 'xs')
                .add_output('ys', 'ys')
        )
        loader = ds.DataLoader(batch_size=params.batch_size,
                               num_workers=params.num_workers,
                               pin_memory=params.pin_memory)

        self.regist_dataloader_with_stage(stage, loader)


# Trainer


class ResnetTrainer(callbacks.TrainCallback,
                    Trainer):
    def icallbacks(self, params: ParamsType):
        callbacks.LoggerCallback(step_frequence=1).hook(self)
        callbacks.LRSchedule().hook(self)  # auto get params.lr_sche to apply lr rate
        callbacks.AutoLoadModel().hook(self)

        if isinstance(self, callbacks.TrainCallback):
            self.hook(self)

    def imodels(self, params: ParamsType):
        super().imodels(params)

        model = resnet34()
        model.fc = nn.Linear(1024, params.n_classes)
        self.model = model
        self.optim = params.optim.build(self.model.parameters())
        self.to_device()

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        meter = Meter()
        xs, ys = batch['xs'], batch['ys']
        logits = self.model(xs)
        loss = F.cross_entropy(logits, ys)

        meter.Lce = loss
        meter.Acc = (logits.argmax(dim=-1) == ys).float().mean()

        self.optim.zero_grad()
        self.accelerator.backward(loss)
        self.optim.step()
        return meter


def main():
    pm = DemoPM()
    pm.from_args()

    dm = CVDM()

    tr = ResnetTrainer(pm)
    tr.train(dm)
    tr.save_model()


if __name__ == '__main__':
    main()
