from lumo.core.interp import *
from torch.optim.sgd import SGD
from torch import nn
from lumo import BaseParams


def test_scheduler_is_attr():
    linear = Linear(0, 1, 0, 100)
    assert isinstance(linear, BaseParams)
    assert linear(0) == 0
    assert linear.get(-1) == 0
    assert linear(-1) == 0
    assert linear(100) == 1
    assert linear(120) == 1
    assert linear.get(120) == 1

    optim = SGD(nn.Linear(10, 10).parameters(), lr=0.3, momentum=0.9)
    for param_group in optim.param_groups:  # type:dict
        assert param_group['lr'] == 0.3

    assert linear.apply(optim, 0) == 0
    for param_group in optim.param_groups:  # type:dict
        assert param_group['lr'] == 0

    assert linear.apply(optim, 100) == 1
    for param_group in optim.param_groups:  # type:dict
        assert param_group['lr'] == 1

    optim = SGD(nn.Linear(10, 10).parameters(), lr=0.3, momentum=0.9)

    assert linear.scale(optim, 0) == 0
    for param_group in optim.param_groups:  # type:dict
        assert param_group['lr'] == 0

    assert linear.scale(optim, 100) == 1
    for param_group in optim.param_groups:  # type:dict
        assert param_group['lr'] == 0.3

    cos = Cos(0, 1, 0, 100)
    assert cos.get(-1) == 0
    assert cos.get(0) == 0
    assert cos.get(50) == 0.5
    assert cos.get(100) == 1
    assert cos.get(101) == 1

    pcos = PeriodCos(0, 1, period=10)
    assert pcos.get(0) == 0
    assert pcos.get(2.5) == 0.5
    assert pcos.get(5) == 1
    assert (pcos.get(7.5) - 0.5) < 1e-5
    assert pcos.get(10) == 0
    assert pcos.get(15) == 1
    assert abs(pcos.get(9.99)) < 1e-5
    # pcos.get(10) == 1
    phcos = PeriodHalfCos(0, 1, period=10)
    assert phcos.get(0) == 0
    assert phcos.get(5) == 0.5
    assert abs(phcos.get(9.99) - 1) < 1e-5
    assert phcos.get(10) == 0
    assert phcos.get(15) == 0.5
