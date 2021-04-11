from typing import Union, NoReturn, Dict, overload, TYPE_CHECKING

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .trainer import ParamsType


class DataModule():
    def __init__(self):
        self._dataloader = {}

    @property
    def dataloaders(self) -> Dict[str, DataLoader]:
        return self._dataloader

    @property
    def train_dataloader(self) -> Union[NoReturn, DataLoader]:
        return self._dataloader.get('train', None)

    @property
    def test_dataloader(self) -> Union[NoReturn, DataLoader]:
        return self._dataloader.get('test', None)

    @property
    def val_dataloader(self) -> Union[NoReturn, DataLoader]:
        return self._dataloader.get('val', None)

    @overload
    def regist_dataloader(self, train: DataLoader = None, val: DataLoader = None, test: DataLoader = None,
                          **others: Dict[str, DataLoader]):
        ...

    def regist_dataloader(self, **kwargs: dict):
        self._dataloader.update(kwargs)

    def idataloader(self, params: 'ParamsType'):
        pass
