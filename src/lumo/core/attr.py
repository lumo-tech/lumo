"""
easydict, support dic['a.b'] to refer dic.a.b
"""
from collections import OrderedDict
from typing import List


class Attr(OrderedDict):

    def __setattr__(self, key: str, value):
        _set_item(self, key.split('.'), value)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Key in attr must be str')
        _set_item(self, key.split('.'), value)

    def __getattr__(self, key: str):
        try:
            res = _get_item(self, key.split('.'))
        except KeyError:
            res = Attr()
            _set_item(self, key.split('.'), res)

        return res

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError('Key in attr must be str')
        return _get_item(self, key.split('.'))


def _set_item(dic, keys: List[str], value):
    if len(keys) == 1:
        if isinstance(value, dict):
            value = dic(value)
        OrderedDict.__setitem__(dic, keys[0], value)
    else:
        nex = Attr()
        OrderedDict.__setitem__(dic, keys[0], nex)
        _set_item(nex, keys[1:], value)


def _get_item(dic, keys: List[str]):
    if len(keys) == 1:
        return OrderedDict.__getitem__(dic, keys[0])
    else:
        nex = OrderedDict.__getitem__(dic, keys[0])
        if isinstance(nex, dict):
            return _get_item(nex, keys[1:])
        else:
            raise KeyError(keys)
