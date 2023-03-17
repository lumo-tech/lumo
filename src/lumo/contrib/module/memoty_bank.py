from torch import distributed
from lumo.proc.dist import is_dist, gather
import torch.distributed

from torch import nn
import torch


class StorageBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = 0
        self.sizes = {}

    def register(self, name, dim, k, dtype=None):
        if dim <= 0:
            bank = (torch.ones(k) + float('inf')).to(dtype=dtype)
        else:
            bank = (torch.ones(k, dim) + float('inf')).to(dtype=dtype)

        self.register_buffer(name, bank)
        self.sizes[name] = bank.shape[0]

    def __getitem__(self, item):
        return self.__getattr__(item)

    @torch.no_grad()
    def scatter(self, name, value, index):
        value = value.detach()
        value = gather(value)
        if isinstance(index, torch.Tensor):
            index = gather(index)
        self[name][index] = value


class MemoryBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = 0

        self.offsets = {}
        self.sizes = {}

    def register(self, name, dim, k, dtype=None):
        if dim <= 0:
            bank = torch.rand(k, dtype=dtype)
        else:
            bank = torch.rand(k, dim, dtype=dtype)
        self.register_buffer(name, bank)
        self.offsets[name] = 0
        self.sizes[name] = k

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattr__(item)

    @torch.no_grad()
    def push(self, name, value):
        if name not in self.offsets or self[name].ndim != value.ndim:
            raise AssertionError()

        value = value.detach()
        value = gather(value)
        ptr = self.offsets[name]
        k = self.sizes[name]
        batch_size = value.shape[0]
        if ptr + batch_size > k:
            batch_size = k - ptr
            value = value[:batch_size]
        self[name][ptr:ptr + batch_size] = value
        self.offsets[name] = (ptr + batch_size) % k


@torch.no_grad()
def batch_shuffle_ddp(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    if not is_dist():
        return x, torch.arange(len(x))
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    if not is_dist():
        return x
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]
