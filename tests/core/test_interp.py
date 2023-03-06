from lumo.core.interp import *
import numpy as np
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


def test_period_linear():
    start = 0
    end = 10
    period = 5
    left = 0
    constant = False
    cur = 2

    expected = 4.0
    result = PeriodLinear.interp(cur, start, end, left, period, constant)
    assert result == expected


def test_power_decay():
    start = 1
    decay_steps = 5
    decay_rate = 0.5
    end = None
    cur = 10

    expected = 0.25
    power_decay = PowerDecay(start, decay_steps, decay_rate, end)
    result = power_decay(cur)
    assert result == expected


def test_power_decay2():
    start = 1
    schedules = [5, 10]
    gammas = [0.5, 0.2]
    cur = 12

    expected = 0.1
    power_decay2 = PowerDecay2(start, schedules, gammas)
    result = power_decay2(cur)
    assert result == expected


# def test_ABCContinuous():
#     # Test ABCContinuous class
#     abc = ABCContinuous(start=1, end=2, left=0, right=10)
#     assert abc(0) == 1
#     assert abc(10) == 2
#     assert abc(5) == abc.interp(5, start=1, end=2, left=0, right=10)


def test_Exp():
    # Test Exp class
    exp = Exp(start=1, end=2, left=0, right=10)
    assert np.isclose(exp(0), 1)
    assert np.isclose(exp(10), 2)
    assert np.isclose(exp(5), 1.078716025, rtol=1e-5)
    assert np.isclose(exp(8), 1.366531851)
    assert np.isclose(exp(9.5), 1.7784638857)


def test_Log():
    # Test Log class
    log = Log(start=1, end=2, left=0, right=10)
    assert np.isclose(log(0), 1)
    assert np.isclose(log(10), 2)
    assert np.isclose(log(5), 1.921283974)


def test_Constant():
    # Test Constant class
    const = Constant(value=0.5)
    assert const(0) == 0.5
    assert const(10) == 0.5
