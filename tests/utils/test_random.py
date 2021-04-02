"""

"""

from lumo.utils.random import *


def test_state():
    state_dict = fix_seed(10)
    a, b, c = random.randint(0, 10), np.random.rand(1), torch.rand(1)
    set_state(state_dict)
    d, e, f = random.randint(0, 10), np.random.rand(1), torch.rand(1)

    assert a == d and b == e and c == f
