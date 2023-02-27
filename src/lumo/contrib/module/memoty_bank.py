import torch
from accelerate.utils import gather
from torch import nn


class MemoryBank(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.offset = 0
        self.k = k

        self.offsets = {}
        self.sizes = {}

    def register(self, name, dim):
        if dim <= 0:
            bank = torch.rand(self.k)
        else:
            bank = torch.rand(self.k, dim)
        self.register_buffer(name, bank)
        self.offsets[name] = 0
        self.sizes[name] = bank.shape[0]

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
