from lumo.trainer.rnd import RndManager
import random
import numpy as np
import torch


def test_random():
    RndManager().mark('123')
    a, b, c = random.randint(0, 1000), np.random.rand(10), torch.rand(10)
    RndManager().mark('123')
    d, e, f = random.randint(0, 1000), np.random.rand(10), torch.rand(10)
    assert a == d and (b == e).all() and (c == f).all()
    RndManager().mark('123')
    RndManager().shuffle()
    d, e, f = random.randint(0, 1000), np.random.rand(10), torch.rand(10)
    assert a != d
    assert (b != e).any()
    assert (c != f).any()

    RndManager().shuffle(2)
    a, b, c = random.randint(0, 1000), np.random.rand(10), torch.rand(10)
    RndManager().shuffle(2)
    d, e, f = random.randint(0, 1000), np.random.rand(10), torch.rand(10)
    assert a == d
    assert (b == e).any()
    assert (c == f).any()
