import torch
from torch import nn
from torch.nn import functional as F
from lumo.core import MetricType
from lumo import Trainer, TrainerParams, callbacks, Meter, TrainStage
from lumo.data import DatasetBuilder

from lumo import DataModule


# from lumo import ParamsType

## define You own Params class
class MyParams(TrainerParams):

    def __init__(self):
        super().__init__()
        self.batch_size = 128
        # Create a optim params dict in Params class for better record.
        self.optim = self.OPTIM.create_optim('SGD', lr=0.1, momentum=0.9, weight_decay=5e-5)


## recommended for IDE smart tips
ParamsType = MyParams


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10),
        )

    def forward(self, xs):
        bs = xs.shape[0]
        xs = xs.reshape(bs, -1)
        return self.mlp(xs)


## Create Your Trainer
class MyTrainer(Trainer):

    ## initialize your callbacks
    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.LoggerCallback(break_in=200).hook(self)

    ## initialize your models (and optimizers)
    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = MyModel()
        self.optim = params.optim.build(self.model.parameters())

        # Create to send model into cuda device assigned in params
        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        meter = Meter()

        xs, ys = batch['xs'], batch['ys']
        logits = self.model(batch['xs'])
        Lall = F.cross_entropy(logits, ys)

        # loss.backward()
        self.accelerate.backward(Lall)
        self.optim.step()

        with torch.no_grad():
            meter.mean.A = (logits.argmax(dim=-1) == ys).float().mean()

        meter.mean.Lall = Lall

        return meter


class MyDataModule(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)
        # Use stage.is_xxx() to decide which type of dataloader you need to create.
        if stage.is_train():
            db = (
                DatasetBuilder()
                    .add_input('xs', torch.rand(5000, 28, 28))
                    .add_input('ys', torch.randint(0, 10, (5000,)))
                    .add_output('xs', 'xs')
                    .add_output('ys', 'ys')
            )
            loader = db.DataLoader(batch_size=params.batch_size)

        else:
            # if stage.is_test()
            # if stage.is_val()
            db = (
                DatasetBuilder()
                    .add_input('xs', torch.rand(500, 28, 28))
                    .add_input('ys', torch.randint(0, 10, (500,)))
                    .add_output('xs', 'xs')
                    .add_output('ys', 'ys')
            )
            loader = db.DataLoader(batch_size=params.batch_size)
        self.regist_dataloader_with_stage(stage, loader)


def main():
    params = MyParams()
    params.from_args()  # see params example for more usage methods.

    # Create a DataModule object to handle all dataloaders
    dm = MyDataModule()

    trainer = MyTrainer(params, dm)

    # Create loader for train
    # trainer.train(loader)
    trainer.train()


if __name__ == '__main__':
    main()
