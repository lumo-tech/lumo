from lumo.core.list import llist
import numpy as np


def test_llist():
    k = llist([1, 2, 3, 4])
    assert k[np.array([0, 1])] == [1, 2]
    assert k[np.array(1)] == 2
