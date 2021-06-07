"""
 - 图 Dataset， gg['item_id'] -> gg['words'] -> gg['item_id'] ，先随机采样，然后尝试直接聚类的方式？
 - Bart 模型 + p-tuning
"""
from lumo import Trainer, Logger, Params, DatasetBuilder, DataModule, K, callbacks
from lumo.contrib.data.collate import CollateBase
from lumo.base_classes import TrainerStage
from lumo.kit import ParamsType


class MultiPM(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 20
        self.batch_size = 128
        self.num_workers = 8
        self.pin_memory = True

        self.pretrain = True
        self.pretrain_path = None

    def iparams(self):
        super().iparams()
        self.lr_sche = self.SCHE.Cos(start=0, end=self.epoch,
                                     left=self.optim.args.lr,
                                     right=1e-6)


class Key(K):
    pass


K = Key


# DataModule
class GlobalTransform():
    def __init__(self, params: MultiPM):
        self.params = params

    def __call__(self, mem: dict):
        return mem


def mtds(params: MultiPM):
    """
    视频路径

    :param params:
    :return:
    """

    ds = (
        DatasetBuilder()
            # .add_input()
            # .add_output()
            .add_global_transform(GlobalTransform(params))
    )

    return ds


class MTCollate(CollateBase):

    def initial(self, params: MultiPM):
        pass


class MTDM(DataModule):

    def idataloader(self, params: MultiPM, stage: TrainerStage, repeat: bool = False):
        ds = mtds(params)

        if stage.is_train:
            ds.subset(range(len(ds) * 4 // 5))
        elif stage.is_eval:
            ds.subset(range(len(ds) * 4 // 5, len(ds)))

        loader = ds.DataLoader(batch_size=params.batch_size,
                               num_workers=params.num_workers,
                               pin_memory=params.pin_memory,
                               collate_fn=CollateBase())

        self.regist_dataloader_with_stage(stage, loader)


# Trainer


class MTTrainer(callbacks.TrainCallback,
                Trainer):
    def icallbacks(self, params: ParamsType):
        callbacks.LoggerCallback(step_frequence=1).hook(self)
        callbacks.LRSchedule().hook(self)  # auto get params.lr_sche to apply lr rate
        callbacks.AutoLoadModel().hook(self)

        if isinstance(self, callbacks.TrainCallback):
            self.hook(self)

    def imodels(self, params: ParamsType):
        super().imodels(params)

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        pass

    def evaluate_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        pass


def main():
    pm = MultiPM()
    pm.from_args()

    dm = MTDM()

    tr = MTTrainer(pm)
    tr.train(dm)


if __name__ == '__main__':
    main()
