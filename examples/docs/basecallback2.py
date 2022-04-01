from lumo import Trainer, Params
from lumo.trainer import callbacks
from lumo import ParamsType


class MyCallback(callbacks.BaseCallback):
    def on_foo_begin(self, source: Trainer, func, params, *args, **kwargs):
        print(source, func, params)

    def on_foo_end(self, source: Trainer, func, params, result, *args, **kwargs):
        print(source, func, params, result)


class MyTrainer(Trainer):
    callback_function = ['foo']

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        MyCallback().hook(self)

    def foo(self):
        return 10


MyTrainer(Params()).foo()
