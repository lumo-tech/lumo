"""
A module to manage dataloaders for different stages of training.
"""
from typing import NoReturn, Union, overload, Optional

from torch.utils.data import DataLoader

from lumo.core import TrainStage, ParamsType
from .loader import DataLoaderType
from .loader import DataLoaderSide


class DataModule:
    """
    Used in Trainer to easily access DataLoaders for different stage(train/test/eval/others).
    """

    def __init__(self, params: ParamsType = None):
        self._prop = {}
        self.params = params

    @property
    def prop(self):
        """
        Get the dictionary of registered dataloaders.

        Returns:
            dict: the dictionary of registered dataloaders.
        """
        return self._prop

    @staticmethod
    def _parse_dataset(loader):
        """
       Parse a dataset from a dataloader.

       Args:
           loader (Union[DataLoader, DataLoaderSide]): a dataloader or a dataloader side.

       Returns:
           Dataset: a dataset if `loader` is a `DataLoader` or a `DataLoaderSide`, None otherwise.

       """
        if isinstance(loader, DataLoader):
            return loader.dataset
        elif isinstance(loader, DataLoaderSide):
            return loader.dataset
        raise NotImplementedError(type(loader))

    @property
    def train_dataset(self):
        """
        Get the train dataset.

        Returns:
            Dataset: the train dataset.

        """
        return self._parse_dataset(self.train_dataloader)

    @property
    def test_dataset(self):
        """Get the test dataset."""
        return self._parse_dataset(self.test_dataloader)

    @property
    def val_dataset(self):
        """Get the validation dataset."""
        return self._parse_dataset(self.val_dataloader)

    @property
    def train_dataloader(self) -> Optional[DataLoaderType]:
        """Get the train dataloader."""
        return self.get_loader_with_stage(TrainStage.train)

    @property
    def test_dataloader(self) -> Optional[DataLoaderType]:
        """Get the test dataloader."""
        return self.get_loader_with_stage(TrainStage.test)

    @property
    def val_dataloader(self) -> Union[NoReturn, DataLoaderType]:
        """Get the val dataloader."""
        return self.get_loader_with_stage(TrainStage.val)

    def get_loader_with_stage(self, stage: TrainStage) -> DataLoaderType:
        """Get the dataloader for a given stage."""
        res = self._prop.get(stage.value, None)
        if res is None:
            self.idataloader(self.params, stage)
            res = self._prop.get(stage.value, None)
        return res

    def __getitem__(self, key):
        return self.prop.get(key, None)

    @overload
    def regist_dataloader(self, train=None, test=None, val=None, **kwargs):
        """
        Registers the given dataloaders under the given keys.

        Args:
            train: A DataLoaderType object for the train set.
            test: A DataLoaderType object for the test set.
            val: A DataLoaderType object for the validation set.
            **kwargs: A DataLoaderType object for other stage
        """

    def regist_dataloader(self, **kwargs: dict):
        """
        Registers the given dataloaders under the given keys.

        Args:
            train: A DataLoaderType object for the train set.
            test: A DataLoaderType object for the test set.
            val: A DataLoaderType object for the validation set.
            **kwargs: A DataLoaderType object for other stage
        """
        for k, v in kwargs.items():
            self.prop[k] = v

    def regist_dataloader_with_stage(self, stage: TrainStage, dl: DataLoaderType):
        """
        Registers the given dataloader under the given TrainStage.

        Args:
            stage: A TrainStage object.
            dl: A DataLoaderType object.
        """
        self.regist_dataloader(**{stage.value: dl})

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        """
        Interface function to implement in a child class to set up data loading.

        Args:
            params: A ParamsType object containing data loading parameters.
            stage: A TrainStage object indicating which stage to set up data loading for.
        """
        pass
