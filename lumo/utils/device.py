from typing import List, Dict, Tuple, Union, overload, Optional
from torch import nn
import torch
from collections.abc import Sequence, Mapping


def _to_device(item: Union[Sequence, Mapping, torch.Tensor, nn.Module],
               device_args_kargs: Tuple[Sequence, Mapping]):
    if isinstance(item, nn.Module):
        return item.to(*device_args_kargs[0], **device_args_kargs[1])

    if isinstance(item, Sequence):
        return [_to_device(ele, device_args_kargs) for ele in item]
    elif isinstance(item, Mapping):
        return {k: _to_device(ele, device_args_kargs) for k, ele in item.items()}
    elif isinstance(item, torch.Tensor):
        return item.to(*device_args_kargs[0], **device_args_kargs[1])
    else:
        raise TypeError(type(item))


def set_default_to_device_func(func):
    global to_device
    to_device = func


to_device = _to_device


def get_to_device_func():
    return to_device


@overload
def construct_device_args_kwargs(self, dtype: torch.dtype, non_blocking: bool = False, copy: bool = False):
    ...


@overload
def construct_device_args_kwargs(self, device: Optional[Union[torch.device, str]] = None,
                                 dtype: Optional[torch.dtype] = None,
                                 non_blocking: bool = False, copy: bool = False):
    ...


@overload
def construct_device_args_kwargs(self, other: torch.Tensor, non_blocking: bool = False, copy: bool = False):
    ...


def construct_device_args_kwargs(*args, **kwargs):
    return (args, kwargs)
