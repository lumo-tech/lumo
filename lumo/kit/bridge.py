"""

"""
from typing import Optional, overload, Union, TypeVar

T = TypeVar('T')

import torch

from lumo.contrib.device import to_device


class DataBridge:
    """
    主要功能：
         - 对输出的数据直接设置 device ，支持对不同的字段设置不同的 device
    """

    def __init__(self, to_device_fn=to_device):
        self._to_device_fn = to_device_fn
        self.device_arg = None
        self.key_device_arg = {}

    @overload
    def to(self, dtype: torch.dtype, non_blocking: bool = False, copy: bool = False):
        ...

    @overload
    def to(self, device: Optional[Union[torch.device, str, int]] = None, dtype: Optional[torch.dtype] = None,
           non_blocking: bool = False, copy: bool = False):
        ...

    @overload
    def to(self, other: torch.Tensor, non_blocking: bool = False, copy: bool = False):
        ...

    def to(self, *args, **kwargs):
        """
        Lazy call for batch data.

        Args: see `torch.Tensor.to` for details
        """
        if len(args) == 0:
            assert 'device' in kwargs
            device = kwargs['device']
        else:
            device = args[0]
        assert isinstance(device, (torch.device, str))
        self.device_arg = [args, kwargs]
        return self

    @overload
    def key_to(self, key, dtype: torch.dtype, non_blocking: bool = False, copy: bool = False):
        ...

    @overload
    def key_to(self, key, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None,
               non_blocking: bool = False, copy: bool = False):
        ...

    @overload
    def key_to(self, key, other: torch.Tensor, non_blocking: bool = False, copy: bool = False):
        ...

    def key_to(self, key, *args, **kwargs):
        """
        When constructing batch data, data with name or index `key` will be assigned/convert to the specific device.

        It has higher priority than `to()` function, and will be called before `to`. Then data with name `key` will be
        ignored when calling `to()` function.
        """
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __call__(self, batch: T) -> T:
        return self._to_device_fn(batch,) # TODO
