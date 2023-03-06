import numpy as np
import torch

from lumo.core.attr import Attr as attr, set_item_iterative


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
    print(res)
    assert (
            str(res) == "Attr([('a', 1), ('nn', None), ('kk', Attr([('k', None)])), ('st', 'NAttr()'), "
                        "('b', [2, 3, 4]), ('c', Attr([('a', 1), ('b', [5, 6, 7]), ('c', Attr([('d', [8, 9])]))])), "
                        "('d', tensor(1.)), ('e', tensor([2., 3., 4.])), ('f', array(2)), ('g', array([2, 3, 4]))])"
    )
    res.update(a=6, b=[4, 5])
    res['c.c.e.f'] = 5
    assert res.a == 6
    assert res.b == [4, 5]
    assert res['c.c.e.f'] == 5
    assert res['c.a'] == 1
    assert res['c.b'] == [5, 6, 7]
    assert isinstance(res['c.c.e'], dict)


def test_get_set():
    res = {}
    set_item_iterative(res, ['a', 'b', 'c'], 4)
    assert isinstance(res['a'], dict)
    assert isinstance(res['a']['b'], dict)
    assert res['a']['b']['c'] == 4
    # set_item_iterative(res, '')
