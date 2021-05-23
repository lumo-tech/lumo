from torchvision.datasets.cifar import CIFAR100
import torch
import numpy as np
import random
import os
from lumo.utils.paths import cache_dir
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

    def image(self, image_size: int = None, return_tensor='np', channle_first=False):
        if image_size is None:
            image_size = self.image_size
        res = random.choice(self.data)  # type: np.ndarray

        if channle_first:
            res = res.transpose(2, 0, 1)
        if return_tensor == 'pt':
            res = torch.from_numpy(res)
        return res

    def images(self, bs=2, resize=None, return_tensor='np'):
        ind = np.random.permutation(len(self.data))[:bs]
        return self.data[ind]
