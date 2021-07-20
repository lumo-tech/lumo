from torchvision.datasets.cifar import CIFAR100
import torch
import numpy as np
import random
import os
from lumo.proc.path import cache_dir
from lumo.utils import cacher
from lumo.kit.logger import get_global_logger

log = get_global_logger()


class MockImage:
    def __init__(self, image_size: int = 32):
        mock_dataset_root = os.path.join(cache_dir(), 'mock')
        cache_args = (MockImage, mock_dataset_root)
        item, path = cacher.load_if_exists(*cache_args)
        self.image_size = image_size
        if item is not None:
            log.dddebug(f'Load MockImage from cache {path}')
            self.data, self.targets = item
        else:
            cifar100 = CIFAR100(root=mock_dataset_root, train=True, download=True)
            self.data, self.targets = cifar100.data, cifar100.targets
            path = cacher.save_cache([cifar100.data, cifar100.targets], *cache_args)
            log.dddebug(f'Save MockImage cache to {path}')

    def image(self, channle_first=False):
        res = random.choice(self.data)  # type: np.ndarray
        if channle_first:
            res = res.transpose(2, 0, 1)
        res = torch.from_numpy(res)

        return res

    def images(self, bs=2, channle_first=False):
        ind = np.random.permutation(len(self.data))[:bs]
        res = self.data[ind]
        if channle_first:
            res = res.transpose(0, 3, 1, 2)
        res = torch.from_numpy(res)

        return res
