from lumo.base_classes import attr
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


def test_jsonify():
    res = get_res()
    jsn = res.jsonify()
    rres = attr().from_dict(jsn)

    assert rres.hash() == res.hash()


def test_copy():
    res = get_res()
    copy = res.copy()

    res.a = 2
    res.b.append(5)
    res.c.a = 2
    res.c.b.append(8)
    res.c.c.d.append(8)

    res.d.uniform_()
    res.e.uniform_()

    assert copy.a == 1
    assert copy.b == [2, 3, 4]
    assert copy.c.a == 1
    assert copy.c.b == [5, 6, 7]
    assert copy.c.c.d == [8, 9]
    assert copy.d == torch.tensor(1)
    assert (copy.e == torch.tensor([2, 3, 4])).all()


def test_deepcopy():
    res = get_res()
    from copy import deepcopy
    copy = deepcopy(res)
    assert res.hash() == copy.hash()


def test_replace():
    res = get_res()
    res.replace(a=6).replace(b=7)
    assert res.a == 6
    assert res.b == 7
