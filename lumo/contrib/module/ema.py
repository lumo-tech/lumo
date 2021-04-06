from torch import nn
import torch
from copy import deepcopy
from torch import LongTensor

if torch.cuda.is_available():
    from torch.cuda import LongTensor as CLongTensor
else:
    CLongTensor = LongTensor

from typing import TypeVar

Module = TypeVar('Module', bound=nn.Module)


def EMA(model: Module, alpha=0.999) -> Module:
    """
    Exponential Moving Average(EMA) for nn.Module
    Args:
        model: nn.Module, An EMA wrapper of original model
        alpha: float, default 0.999, decay ratio of EMA

    Returns:
        A new cloned model that has a new method 'step'

    Notes:
        ema model will not generate gradient in its forward process
    """
    ema_model = deepcopy(model)
    [i.detach_() for i in ema_model.parameters()]

    def step(alpha_=None):

        if alpha_ is None:
            alpha_ = alpha

        with torch.no_grad():
            for (_, ema_param), (_, param) in zip(ema_model.state_dict().items(), model.state_dict().items()):
                ema_param.to(param.device)
                if not isinstance(param, (LongTensor, CLongTensor)):
                    ema_param.data.mul_(alpha_).add_(param.data, alpha=1 - alpha_)
                else:
                    ema_param.data.copy_(alpha_ * ema_param + (1 - alpha_) * param)

    forward_ = ema_model.forward

    def forward(*args, **kwargs):
        with torch.no_grad():
            return forward_(*args, **kwargs)

    ema_model.forward = forward
    ema_model.step = step
    return ema_model
