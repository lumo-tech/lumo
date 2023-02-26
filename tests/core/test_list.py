from lumo.core.list import llist
import torch
import numpy as np


def test_llist():
    k = llist([1, 2, 3, 4])
    # get item
    assert k[1:2] == [2]

    assert k[np.array(1)] == 2
    assert k[torch.tensor(1)] == 2

    # get slice
    assert k[np.array([0, 1])] == [1, 2]
    assert k[torch.tensor([1, 2])] == [2, 3]
    assert k[[2, 3]] == [3, 4]

    # from mask
    assert k[np.array([True, True, False, False])] == [1, 2]
    assert k[np.array([False, True, True])] == [2, 3]

    # index error
    try:
        _ = k[4]
        assert False
    except IndexError:
        pass
