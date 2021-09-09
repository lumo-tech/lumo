import numpy as np
import torch


def to_ndarray(item):
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu()
    return np.array(item)
