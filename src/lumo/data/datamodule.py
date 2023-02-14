from typing import NoReturn, Union, overload, Optional

from torch.utils.data import DataLoader

from lumo.core import PropVar, ParamsType, TrainStage
from .loader import DataLoaderType
from .loader import DataLoaderSide


class DataModule(metaclass=PropVar):
    def __init__(self, params: ParamsType = None):
        self.params = params

    @staticmethod
    def _parse_dataset(loader):
        if isinstance(loader, DataLoader):
            return loader.dataset
        elif isinstance(loader, DataLoaderSide):
            return loader.dataset
        return None

    @property
    def train_dataset(self):
        return self._parse_dataset(self.train_dataloader)

    @property
    def test_dataset(self):
        return self._parse_dataset(self.test_dataloader)

    @property
    def val_dataset(self):
        return self._parse_dataset(self.val_dataloader)

    @property
    def train_dataloader(self) -> Optional[DataLoaderType]:
        return self.get_loader_with_stage(TrainStage.train)

    @property
    def test_dataloader(self) -> Optional[DataLoaderType]:
        return self.get_loader_with_stage(TrainStage.test)

    @property
    def val_dataloader(self) -> Union[NoReturn, DataLoaderType]:
        return self.get_loader_with_stage(TrainStage.val)

    def get_loader_with_stage(self, stage: TrainStage) -> DataLoaderType:
        res = self._prop.get(stage.value, None)
        if res is None:
            self.idataloader(self.params, stage)
            res = self._prop.get(stage.value, None)
        return res

    def __getitem__(self, key):
        return self._prop.get(key, None)

    @overload
    def regist_dataloader(self, train=None, test=None, val=None):
        ...

    def regist_dataloader(self, **kwargs: dict):
        for k, v in kwargs.items():
            self._prop[k] = v

    def regist_dataloader_with_stage(self, stage: TrainStage, dl: DataLoaderType):
        self._prop[stage.value] = dl

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        pass
