from typing import Union, NoReturn, overload, Dict
from torch.utils.data import DataLoader
from lumo.base_classes import TrainerStage
from lumo.kit.params import ParamsType


class DataModuleMix():
    @property
    def train_dataloader(self) -> Union[NoReturn, DataLoader]:
        raise NotImplementedError()

    @property
    def test_dataloader(self) -> Union[NoReturn, DataLoader]:
        raise NotImplementedError()

    @property
    def val_dataloader(self) -> Union[NoReturn, DataLoader]:
        raise NotImplementedError()

    @overload
    def regist_dataloader(self, train: DataLoader = None, val: DataLoader = None, test: DataLoader = None,
                          **others: Dict[str, DataLoader]):
        ...

    def regist_dataloader(self, **kwargs: dict):
        raise NotImplementedError()

    def idataloader(self, params: ParamsType, stage: TrainerStage, repeat: bool = False):
        raise NotImplementedError()

    def iidataloader(self, params: ParamsType, stage: TrainerStage, repeat: bool = False):
        if not repeat:
            self.idataloader(params=params, stage=stage, repeat=repeat)



class BaseCallbackMix():
    def icallbacks(self, params: ParamsType):
        raise NotImplementedError()


class CallbackMix(BaseCallbackMix):

    def add_callback(self, callback):
        raise NotImplementedError()

    def reload_callback(self, callback):
        raise NotImplementedError()

    def remove_callback(self, callback):
        raise NotImplementedError()


class ModelMix():

    def imodels(self, params: ParamsType):
        raise NotImplementedError()

    def ioptims(self, params: ParamsType):
        raise NotImplementedError()

    def optim_state_dict(self):
        raise NotImplementedError()

    def model_state_dict(self):
        raise NotImplementedError()
