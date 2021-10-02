from lumo.kit.random import Random
from lumo.utils.random import *


def test_random():
    Random().mark('123')
    a, b, c = random.randint(0, 10), np.random.rand(1), torch.rand(1)
    Random().mark('123')
    d, e, f = random.randint(0, 10), np.random.rand(1), torch.rand(1)
    assert a == d and b == e and c == f
    Random().mark('123')
    Random().shuffle()
    d, e, f = random.randint(0, 10), np.random.rand(1), torch.rand(1)
    assert a != d
    assert b != e
    assert c != f
