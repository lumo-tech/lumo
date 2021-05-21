from lumo.kit.mixin import BaseCallbackMix
from lumo.kit import callbacks
from lumo.kit.params import ParamsType


class BaseCallbackMixin(BaseCallbackMix):
    def icallbacks(self, params: ParamsType):
        params.step_frequence = params.default(1)
        callbacks.LoggerCallback(step_frequence=params.step_frequence).hook(self)
        callbacks.LRSchedule().hook(self)  # auto get params.lr_sche to apply lr rate
        callbacks.AutoLoadModel().hook(self)

        if isinstance(self, callbacks.TrainCallback):
            self.hook(self)
