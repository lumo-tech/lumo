from typing import Tuple

from torch import nn


class Container:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        for i, v in enumerate(self.args):
            yield f'{i}', v

        for k, v in self.kwargs.items():
            yield k, v

    def values(self):
        for k, v in self:
            return v

    def items(self):
        yield from self

    def keys(self):
        for k, v in self:
            yield k


class Losses(Container): pass


class Meters(Container): pass


class FitModel(nn.Module):
    def feed(self, batch, loss_fn) -> Tuple[Losses, Meters]:
        return Losses(), Meters()

    def eval_feed(self, batch, loss_fn) -> Tuple[Losses, Meters]:
        return Losses(), Meters()
