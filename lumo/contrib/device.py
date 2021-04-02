"""

"""
from typing import Union, List, Dict, Tuple

import torch


def to_device(batch: Union[List, Dict, Tuple, torch.Tensor], device_args_kargs):
    if isinstance(batch, (list, tuple)):
        return [to_device(ele, device_args_kargs) for ele in batch]
    elif isinstance(batch, dict):
        return {k: to_device(ele, device_args_kargs) for k, ele in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(*device_args_kargs[0], **device_args_kargs[1])
