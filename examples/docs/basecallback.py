from lumo import Trainer
from lumo.trainer import callbacks
from lumo import ParamsType


class MyTrainer(Trainer):
    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.LoggerCallback().hook(self)
        callbacks.EvalCallback(eval_per_epoch=5, test_per_epoch=10).hook(self)
        callbacks.AutoLoadModel().hook(self)
        callbacks.ScalarRecorder().hook(self)
