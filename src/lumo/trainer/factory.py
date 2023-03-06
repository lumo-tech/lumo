import importlib
from typing import overload

from torch.optim import Optimizer

from lumo.core import interp
from lumo.core.params import BaseParams


class InterpFactory:
    """A factory class for creating instances of various interpolation classes.

    This class provides convenient access to various interpolation classes that are defined in the `interp` module.

    Attributes:
        Cos (class): An interpolation class for cosine interpolation.
        Linear (class): An interpolation class for linear interpolation.
        Exp (class): An interpolation class for exponential interpolation.
        Log (class): An interpolation class for logarithmic interpolation.
        Constant (class): An interpolation class for constant interpolation.
        PeriodCos (class): An interpolation class for periodic cosine interpolation.
        PeriodHalfCos (class): An interpolation class for periodic half-cosine interpolation.
        PeriodTriangle (class): An interpolation class for periodic triangle interpolation.
        PeriodLinear (class): An interpolation class for periodic linear interpolation.
        PowerDecay (class): An interpolation class for power-decay interpolation.
        List (class): An interpolation class for list interpolation.

    """
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
    """A class for building an optimizer with specified parameters.

    Attributes:
        None

    Methods:
        from_kwargs(cls, **kwargs): Creates a new instance of OptimBuilder class and updates its attributes with the given keyword arguments.
        build(self, parameters, optim_cls=None) -> Optimizer: Builds and returns an optimizer with the specified parameters.

    """

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Creates a new instance of OptimBuilder class and updates its attributes with the given keyword arguments.

        Args:
            **kwargs: A dictionary containing the optimizer parameters.

        Returns:
            self: A new instance of the OptimBuilder class with updated attributes.
        """
        self = cls()
        self.update(kwargs)
        return self

    def build(self, parameters, optim_cls=None) -> Optimizer:
        """Builds and returns an optimizer with the specified parameters.

        Args:
            parameters: The parameters for the optimizer.
            optim_cls: The class of the optimizer to be built.

        Returns:
            optim_cls: The built optimizer.

        Raises:
            ModuleNotFoundError: If the specified optimizer class cannot be found in the corresponding module.
        """
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
            if optim_cls is None:
                raise ModuleNotFoundError("Cannot find {} in {}".format(name, optim_lib))

        return optim_cls(parameters, **args)


class _OptimFactory:
    """
        A factory class that provides different optimization algorithms to be used during training.

    Methods:
        create_optim(name=None, **kwargs) -> OptimBuilder:
            Creates an instance of OptimBuilder for a specified optimization algorithm.

    Examples:
        To create an instance of OptimBuilder for Adam optimizer with default values:
        >>> optim_builder = OptimFactory.create_optim(name='Adam')

        To create an instance of OptimBuilder for SGD optimizer with specific values:
        >>> optim_builder = OptimFactory.create_optim(name='SGD', lr=0.01, momentum=0.9)

    """

    @overload
    def create_optim(self, name='SGD', lr=None, momentum=0, dampening=0, weight_decay=0,
                     nesterov=False) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the SGD optimizer."""

    @overload
    def create_optim(self, name='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                     amsgrad=False) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the Adam optimizer."""

    @overload
    def create_optim(self, name='Adadelta', lr=1.0, rho=0.9, eps=1e-6, weight_decay=0) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the Adadelta optimizer."""

    @overload
    def create_optim(self, name='Adagrad', lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
                     eps=1e-10) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the Adagrad optimizer."""

    @overload
    def create_optim(self, name='AdamW', lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the AdamW optimizer."""

    @overload
    def create_optim(self, name='AdamW',
                     lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the AdamW optimizer."""

    @overload
    def create_optim(self, name='ASGD',
                     lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the ASGD optimizer."""

    @overload
    def create_optim(self, name='LBFGS',
                     lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
                     line_search_fn=None) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the LBFGS optimizer."""

    @overload
    def create_optim(self, name='RMSprop', lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0,
                     centered=False) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the RMSprop optimizer."""

    @overload
    def create_optim(self, name='Rprop', lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the Rprop optimizer."""

    @overload
    def create_optim(self, name='SparseAdam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -> OptimBuilder:
        """Creates an instance of OptimBuilder for the SparseAdam optimizer."""

    def create_optim(self, name=None, **kwargs) -> OptimBuilder:
        """Create."""
        return OptimBuilder.from_kwargs(name=name, **kwargs)


OptimFactory = _OptimFactory()
