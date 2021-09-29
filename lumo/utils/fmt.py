import numpy as np
import torch


def to_ndarray(item):
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu()
    return np.array(item)


def detach(item):
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu().numpy()
    return item


def validate_scalar_shape(ndarray, name=''):
    if ndarray.ndim != 0:
        raise ValueError(
            "Expected scalar value for %r but got %r" % (name, ndarray)
        )
    return ndarray


def is_scalar(ndarray):
    if ndarray.ndim != 0:
        return False
    return True
