import random
import numpy as np
import torch
from lumo.utils.random import fix_seed, set_state, hashseed


def test_device():
    import torch
    if torch.cuda.is_available():
        fix_seed(1)
        a = torch.rand(10, device='cuda')
        fix_seed(1)
        d = torch.rand(10, device='cuda')
        assert (a == d).all()

    if torch.has_mps:
        # [2023.02.22] Currently (as MPS support is quite new) there is no way to set the seed for MPS directly.
        # fix_seed(1)
        # a = torch.rand(10, device='mps')
        # fix_seed(1)
        # d = torch.rand(10, device='mps')
        # assert (a == d).all()
        pass


def test_state():
    state_dict = fix_seed(10)
    a, b, c = random.randint(0, 10), np.random.rand(10), torch.rand(10)
    set_state(state_dict)
    d, e, f = random.randint(0, 10), np.random.rand(10), torch.rand(10)

    assert a == d and (b == e).all() and (c == f).all()


def test_hashseed():
    state_dict = fix_seed(hashseed(1))
    a, b, c = random.randint(0, 10), np.random.rand(10), torch.rand(10)
    set_state(state_dict)
    d, e, f = random.randint(0, 10), np.random.rand(10), torch.rand(10)

    assert a == d and (b == e).all() and (c == f).all()


def test_hashseed2():
    fix_seed(hashseed(1))
    a, b, c = random.randint(0, 10), np.random.rand(10), torch.rand(10)
    fix_seed(hashseed(1))
    d, e, f = random.randint(0, 10), np.random.rand(10), torch.rand(10)

    assert a == d and (b == e).all() and (c == f).all()
