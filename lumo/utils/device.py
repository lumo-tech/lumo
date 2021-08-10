from typing import List, Dict, Tuple, Union, overload, Optional, Sequence
from decorator import decorator
from torch import nn
import torch
from collections.abc import Sequence, Mapping

from torch.utils.data import DataLoader


def _to_device(item: Union[Sequence, Mapping, torch.Tensor, nn.Module],
               device_args_kargs: Tuple[Sequence, Mapping]):
    """
    Recursively sends the elements in the item contains tensor/module to a given device.

    Args:
        item: any data structur contains tensor or module.
        device_args_kargs: device argrument

    Returns:

    """
    if isinstance(item, nn.Module):
        return item.to(*device_args_kargs[0], **device_args_kargs[1])

    if isinstance(item, Sequence):
        if isinstance(item, str):
            return item
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


def get_to_device_func():
    return to_device


def to_device_enumrate(loader: DataLoader, device_args_kwargs: Tuple[Sequence, Dict]):
    to_device = get_to_device_func()
    for idx, batch in enumerate(loader):
        batch = to_device(batch, device_args_kwargs)
        yield idx, batch


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


try:
    from accelerate.data_loader import send_to_device
except:
    send_to_device = None


@decorator
def _wrap(func, item, device_arg):
    args, kwargs = device_arg
    device = kwargs.get('device', None)
    if device is None:
        device = args[0]
    return func(item, device)


if send_to_device is not None:
    to_device = _wrap(send_to_device)
else:
    to_device = _to_device
