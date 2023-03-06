import json
import tempfile

from lumo import BaseParams
from lumo.core.raises import BoundCheckError


def test_params():
    params = BaseParams()
    params.limited_value = params.arange(1, 0, 2)  # default, left, right
    params.limited_choice = params.choice('cifar10', 'cifar100', 'imagenet')
    assert params.limited_value == 1
    assert params.limited_choice == 'cifar10'
    dic = params.to_dict()
    assert dic['limited_value'] == 1
    assert dic['limited_choice'] == 'cifar10'
    try:
        params.limited_choice = 'stl10'
        assert False, 'should raise BoundCheckError'
    except BoundCheckError as e:
        assert True

    try:
        params.limited_value = 3
        assert False, 'should raise BoundCheckError'
    except BoundCheckError as e:
        assert True


class MyParams(BaseParams):

    def __init__(self):
        super().__init__()
        self.k = None


def get_res():
    res = BaseParams()
    res.a = 1
    res.nn = None
    res.kk = MyParams()
    res.st = 'NAttr()'
    res.b = [2, 3, 4]
    res.c = {'a': 1, 'b': [5, 6, 7], 'c': {'d': [8, 9]}}
    return res


def test_argv():
    params = get_res()
    params.from_args(['--a', '1', '--d.c.d=2', '--kk.c=3'])
    print(params)
    assert params.a == 1
    assert params.d.c.d == 2
    assert isinstance(params.kk, MyParams)
    assert params.kk.c == 3
    assert isinstance(params.d.c, BaseParams)


def test_dict():
    res = get_res()
    jsn = res.to_dict()
    rres = BaseParams().from_dict(jsn)
    assert rres.hash() == res.hash()


def test_json():
    res = get_res()
    fn = tempfile.mktemp()
    with open(fn, 'w') as w:
        json.dump({'c': {'a': 2}}, w)
    res.from_json(fn)
    assert res.c.a == 2


def test_copy():
    res = get_res()
    copy = res.copy()

    res.a = 2
    res.b.append(5)
    res.c.a = 2
    res.c.b.append(8)
    res.c.c.d.append(8)

    assert copy.a == 1
    assert copy.b == [2, 3, 4]
    assert copy.c.a == 1
    assert copy.c.b == [5, 6, 7]
    assert copy.c.c.d == [8, 9]


def test_copy2():
    res = get_res()
    copy = res.copy()

    print(type(copy))
    copy.c.a = 2
    assert res.c.a == 1
    copy.c.a = 1
    assert res.hash() == copy.hash()
