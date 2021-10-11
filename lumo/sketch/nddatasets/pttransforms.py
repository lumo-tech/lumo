"""

"""
import random

import torch
from torch.nn import functional as F


class RandomHorizontalFlip():
    """Flip randomly the image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x: torch.Tensor):
        """NCWH"""
        if random.random() < self.p:
            x = x.flip(dims=(-1,))
        return x.contiguous()


class RandomBatchCrop():
    def __init__(self, size, padding=4, fill=0, p=0.5):
        self.p = p
        self.size = size
        self.pad = Pad(padding, fill=fill)

    def __call__(self, x: torch.Tensor):
        """NCWH"""
        x = self.pad(x)

        h, w = x.shape[-2:]
        new_h = new_w = self.size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        x = x[:, :, top: top + new_h, left: left + new_w]
        return x


class Pad():
    def __init__(self, padding=4, fill=0):
        self.pad = padding
        self.fill = fill

    def __call__(self, x: torch.Tensor):
        x = F.pad(x, [self.pad] * 4)
        return x
