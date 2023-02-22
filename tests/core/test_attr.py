from lumo.core.attr import Attr as attr
import numpy as np
import torch


class NAttr(attr):

    def __init__(self) -> None:
        super().__init__()
        self.k = None


def get_res():
    res = attr()
    res.a = 1
    res.nn = None
    res.kk = NAttr()
    res.st = 'NAttr()'
    res.b = [2, 3, 4]
    res.c = {'a': 1, 'b': [5, 6, 7], 'c': {'d': [8, 9]}}
    res.d = torch.tensor(1).float()
    res.e = torch.tensor([2, 3, 4]).float()
    res.f = np.array(2)
    res.g = np.array([2, 3, 4])
    return res


def test_replace():
    res = get_res()
    res.update(a=6, b=7)
    assert res.a == 6
    assert res.b == 7
