from lumo import DataModule, ParamsType, TrainStage, DatasetBuilder
import torch


class MyDataModule(DataModule):
    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        if stage.is_train():
            print("init train dataloader")
            db = (
                DatasetBuilder()
                    .add_input("xs", torch.rand(500, 28, 28))
                    .add_input("ys", torch.randint(0, 10, (500,)))
                    .add_output("xs", "xs")
                    .add_output("ys", "ys")
            )
            loader = db.DataLoader(batch_size=10)
        else:
            print("init test dataloader")
            db = (
                DatasetBuilder()
                    .add_input("xs", torch.rand(50, 28, 28))
                    .add_input("ys", torch.randint(0, 10, (50,)))
                    .add_output("xs", "xs")
                    .add_output("ys", "ys")
            )
            loader = db.DataLoader(batch_size=10)
        self.regist_dataloader_with_stage(stage, loader)


dm = MyDataModule()
print(dm._prop)
print(len(dm.train_dataloader))
print(dm._prop)
print(len(dm.test_dataloader))
print(dm._prop)
