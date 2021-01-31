from thexp import Trainer
from trainers import GlobalParams
from thexp.contrib.data import splits
from thexp import DatasetBuilder

from data.transforms import ToTensor
from data.dataxy import datasets

toTensor = ToTensor()


class DatasetMixin(Trainer):
    def datasets(self, params: GlobalParams):
        raise NotImplementedError()


class BaseSupDatasetMixin(DatasetMixin):
    """Base Supervised Dataset"""

    def datasets(self, params: GlobalParams):
        dataset_fn = datasets[params.dataset]

        test_x, testy = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        train_idx, val_idx = splits.train_val_split(train_y,
                                                    val_size=params.val_size)

        test_dataloader = (
            DatasetBuilder(test_x, testy)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size,
                            num_workers=params.num_workers)
        )

        train_dataloader = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=toTensor).add_y()
                .subset(train_idx)
                .DataLoader(batch_size=params.batch_size,
                            num_workers=params.num_workers,
                            shuffle=True)
        )

        val_datalaoder = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=toTensor).add_y()
                .subset(val_idx)
                .DataLoader(batch_size=params.batch_size,
                            num_workers=params.num_workers)

        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_datalaoder,
                                test=test_dataloader)

        self.to(self.device)
