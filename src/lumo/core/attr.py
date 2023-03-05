"""
easydict, support dic['a.b'] to refer dic.a.b
"""
from collections import OrderedDict
from typing import List


class Attr(OrderedDict):

    def __setattr__(self, key: str, value):
        set_item_iterative(self, key.split('.'), value)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Key in attr must be str')
        set_item_iterative(self, key.split('.'), value)

    def __getattr__(self, key: str):
        try:
            res = get_item_iterative(self, key.split('.'))
        except KeyError:
            res = Attr()
            set_item_iterative(self, key.split('.'), res)

        return res

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError('Key in attr must be str')
        return get_item_iterative(self, key.split('.'))


def safe_update_dict(src: dict, kwargs: dict, assert_type=True):
    for ks, v in walk_dict(kwargs):
        try:
            old_v = get_item_iterative(src, ks)
            if old_v is None or isinstance(old_v, type(v)):
                set_item_iterative(src, ks, v)
                # print(ks, v)
            else:
                raise TypeError(ks, type(old_v), type(v))
        except KeyError:
            set_item_iterative(src, ks, v)
            # print(ks, v)
    return src


def walk_dict(dic: dict, root=None):
    if root is None:
        root = []
    for k, v in dic.items():
        if isinstance(v, dict):
            yield from walk_dict(v, [*root, *k.split('.')])
        else:
            yield [*root, *k.split('.')], v


def set_item_iterative(dic: dict, keys: List[str], value):
    if len(keys) == 1:
        if isinstance(value, dict):
            for ks, v in walk_dict(value):
                set_item_iterative(dic, [*keys, *ks], v)
        else:
            dict.__setitem__(dic, keys[0], value)
    else:
        try:
            nex = dict.__getitem__(dic, keys[0])
            if not isinstance(nex, dict):
                raise ValueError(keys[0], nex)
            # dict.__setitem__(dic, keys[0], nex)
        except KeyError:
            nex = dict()
            dict.__setitem__(dic, keys[0], nex)

        set_item_iterative(nex, keys[1:], value)


def get_item_iterative(dic: dict, keys: List[str]):
    if len(keys) == 1:
        return dict.__getitem__(dic, keys[0])
    else:
        nex = dict.__getitem__(dic, keys[0])
        if isinstance(nex, dict):
            return get_item_iterative(nex, keys[1:])
        else:
            raise KeyError(keys)
