from lumo.core.interp import Linear
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
