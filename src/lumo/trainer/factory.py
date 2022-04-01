import importlib
from typing import overload

from torch.optim import Optimizer

from lumo.core import interp
from lumo.core.params import BaseParams


class InterpFactory:
    Cos = interp.Cos
    Linear = interp.Linear
    Exp = interp.Exp
    Log = interp.Log
    Constant = interp.Constant
    PeriodCos = interp.PeriodCos
    PeriodHalfCos = interp.PeriodHalfCos
    PeriodTriangle = interp.PeriodTriangle
    PeriodLinear = interp.PeriodLinear
    PowerDecay = interp.PowerDecay
    List = interp.InterpolateList


class OptimBuilder(BaseParams):

    @classmethod
    def from_kwargs(cls, **kwargs):
        self = cls()
        self.update(kwargs)
        return self

    def build(self, parameters, optim_cls=None) -> Optimizer:
        res = self.copy()
        name = res['name']
        lname = name.lower()
        args = res.copy()
        args.pop('name')

        if optim_cls is None:
            if lname in {'lars'}:
                optim_lib = importlib.import_module("lumo.contrib.optim.{}".format(lname))
            else:
                optim_lib = importlib.import_module("torch.optim.{}".format(lname))
            optim_cls = getattr(optim_lib, name, None)

        return optim_cls(parameters, **args)


class _OptimFactory:
    @overload
    def create_optim(self, name='SGD', lr=None, momentum=0, dampening=0, weight_decay=0,
                     nesterov=False) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                     amsgrad=False) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='Adadelta', lr=1.0, rho=0.9, eps=1e-6, weight_decay=0) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='Adagrad', lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
                     eps=1e-10) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='AdamW', lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='AdamW',
                     lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='ASGD',
                     lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='LBFGS',
                     lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
                     line_search_fn=None) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='RMSprop', lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0,
                     centered=False) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='Rprop', lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)) -> OptimBuilder:
        pass

    @overload
    def create_optim(self, name='SparseAdam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -> OptimBuilder:
        pass

    def create_optim(self, name=None, **kwargs) -> OptimBuilder:
        return OptimBuilder.from_kwargs(name=name, **kwargs)


OptimFactory = _OptimFactory()
