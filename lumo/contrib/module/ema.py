""""""
import warnings

from torch import nn
import torch
from copy import deepcopy
from typing import TypeVar

Module = TypeVar('Module', bound=nn.Module)


def empty_train(*args, **kwargs):
    return args[0]


import numpy as np

np.random.rand()


def EMA(model: Module, alpha=0.999, force_eval=False) -> Module:
    """
    EMA(model: Module, alpha=0.999, force_eval=False)

        An Exponential Moving Average(EMA) wrapper for nn.Module

        Examples:
        --------
        >>> model = ...
        >>> ema_model = EMA(model,alpha=0.999)

            ...

            ema_model.step()
            # or ema_model.step(0.99)

        Args:
            model:
            alpha:
            force_eval:

        Returns:

    """
    ema_model = deepcopy(model)

    [i.requires_grad_(False) for i in ema_model.parameters()]
    param_keys = set([k for k, _ in ema_model.named_parameters()])
    buffer_keys = set([k for k, _ in ema_model.named_buffers()])  # for Norm layers

    def step(alpha_=None):

        if alpha_ is None:
            alpha_ = alpha

        with torch.no_grad():
            for (k, ema_param), (_, param) in zip(ema_model.state_dict().items(), model.state_dict().items()):
                if k in param_keys:
                    ema_param.data.copy_(alpha_ * ema_param + (1 - alpha_) * param)
                elif k in buffer_keys:
                    ema_param.data.copy_(param)

    forward_ = ema_model.forward

    def forward(*args, **kwargs):
        with torch.no_grad():
            return forward_(*args, **kwargs)

    ema_model.forward = forward

    if hasattr(ema_model, 'step'):
        warnings.warn(f'EMA use `step` function to update module paramters, '
                      f'the defined `step` function in class {model.__class__.__name__} will be replaced.')

    ema_model.step = step
    ema_model.eval()
    if force_eval:
        ema_model.train = empty_train
    return ema_model
