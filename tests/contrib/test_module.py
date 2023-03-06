from lumo.contrib.module.memoty_bank import MemoryBank
from accelerate import Accelerator
import torch
import pytest


def test_memory_bank():
    bank = MemoryBank()
    bank.register('test', 32, 512)
    res = [torch.rand(128, 32) for i in range(8)]
    acce = Accelerator()
    for r in res:
        bank.push('test', r)

    assert (bank['test'] == torch.cat(res[4:], dim=0)).all()
    bank.requires_grad_(False)
    # assert (bank['test'][:128] == res[-1]).all()
    # assert (bank['test'][128:] == torch.cat(res[1:4], dim=0)).all()
