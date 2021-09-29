"""

"""
import copy
import warnings
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import torch
from joblib import hash

_attr_clss = {}

ATTR_TYPE = '_type'


class meta_attr(type):
    """记录所有attr子类的类名和类信息，用于序列化和反序列化"""

    def __new__(mcs, *args: Any, **kwargs: Any):
        mcs = type.__new__(mcs, *args, **kwargs)
        _attr_clss[mcs.__name__] = mcs
        return mcs


def _get_item(dic, key: str, default=None):
    if isinstance(key, str):
        for sub in key.split('.'):
            if isinstance(dic, dict) and OrderedDict.__contains__(dic, sub):
                dic = OrderedDict.__getitem__(dic, sub)
            else:
                return default
        return dic
    else:
        if OrderedDict.__contains__(dic, key):
            return OrderedDict.__getitem__(dic, key)
        else:
            return default


def _set_item(dic: dict, key: str, value, overwrite=True):
    if isinstance(key, str):
        subs = key.split('.', maxsplit=1)
        if len(subs) == 1:
            OrderedDict.__setitem__(dic, key, value)
            # dic[key] = value
            return True

        sub, right = subs
        if sub in dic:
            nxt = dic[sub]
            if not isinstance(nxt, dict):
                if overwrite:
                    dic[sub] = attr()
                    nxt = dic[sub]
                else:
                    return False
        else:
            dic[sub] = attr()
            nxt = dic[sub]
        return _set_item(nxt, right, value, overwrite)
    else:
        OrderedDict.__setitem__(dic, key, value)
        return True


class attr(OrderedDict, metaclass=meta_attr):
    """
    An ordered defaultdict, the default class is attr itself.
    """

    @staticmethod
    def __parse_value(v):
        if isinstance(v, attr):
            pass
        elif isinstance(v, dict):
            v = attr.from_dict(v)
        return v

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, name: str, value: Any) -> None:
        value = self.__parse_value(value)
        self[name] = value

    def __getitem__(self, k):
        return _get_item(self, k, attr())

    def __setitem__(self, k, v):
        v = self.__parse_value(v)
        _set_item(self, k, v, True)

    def __contains__(self, o: str) -> bool:
        return not isinstance(_get_item(self, o, _empty()), _empty)

    def __copy__(self):
        return self.from_dict(self)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, dict):
            return self.hash() == hash(o)
        return False

    def items(self):
        return super().items()

    def jsonify(self) -> dict:
        """
        Return a serialized dict which can be dumped in json format.
        You can deserialize this dict by `from_dict()`
        """
        import numbers
        res = dict()
        if type(self) != attr:
            res[ATTR_TYPE] = self.__class__.__name__

        for k, v in self.items():
            if isinstance(v, (numbers.Number, str)):
                res[k] = v
            elif isinstance(v, attr):
                v = v.jsonify()
                res[k] = v
            elif isinstance(v, Iterable):
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    nv = _arr(v).jsonify()
                else:
                    nv = []
                    for vv in v:
                        if isinstance(vv, (numbers.Number, str)):
                            nv.append(vv)
                        elif isinstance(vv, dict):
                            if isinstance(vv, attr):
                                nv.append(vv.jsonify())
                            else:
                                nv.append(attr.from_dict(vv).jsonify())
                        elif vv is None:
                            nv.append(_none().jsonify())
                        else:  # set,list
                            _tmp = attr()
                            _tmp['tmp'] = vv
                            nv.append(_tmp.jsonify()['tmp'])
                res[k] = nv
            elif v is None:
                res[k] = _none().jsonify()

        return res

    def hash(self) -> str:
        """return a hash string, both order/key/value changes will change the hash value."""
        return hash(self.jsonify())

    def copy(self):
        """deep copy"""
        return self.__copy__()

    @classmethod
    def from_dict(cls, dic: dict):
        # res = cls()
        cls_name = dic.get(ATTR_TYPE, cls.__name__)
        if cls_name not in _attr_clss:
            res = attr()
            from lumo.base_classes.errors import AttrTypeNotFoundWarning
            warnings.warn('{} not found, will use class attr to receive values.'.format(cls_name),
                          AttrTypeNotFoundWarning)
        else:
            res = _attr_clss[cls_name].__new__(_attr_clss[cls_name])

        if ATTR_TYPE in dic:
            dic.pop(ATTR_TYPE)

        if isinstance(res, _none):
            return None
        elif isinstance(res, _arr):
            if dic['torch']:
                return torch.tensor(dic['arr'])
            else:
                return np.array(dic['arr'])

        for k, v in dic.items():
            if isinstance(v, dict):
                v = attr.from_dict(v)
            elif isinstance(v, (list, tuple, torch.Tensor, np.ndarray)):
                v = cls._copy_iters(v)

            res[k] = v
        return res

    @staticmethod
    def _copy_iters(item):
        if isinstance(item, torch.Tensor):
            return item.clone()
        elif isinstance(item, np.ndarray):
            return item.copy()
        elif isinstance(item, list):
            return list(attr._copy_iters(i) for i in item)
        elif isinstance(item, set):
            return set(attr._copy_iters(i) for i in item)
        elif isinstance(item, tuple):
            return tuple(attr._copy_iters(i) for i in item)
        elif isinstance(item, dict):
            return attr.from_dict(item)
        return copy.copy(item)


class _none(attr):
    """
    A tricky item used in attr to replace None value when serializing.
    """


class _empty(): pass


class _arr(attr):
    """
    A tricky item used in attr to parse pytorch and numpy array.
    """

    def __init__(self, arr):
        super().__init__()
        self.arr = arr.tolist()
        self.torch = isinstance(arr, torch.Tensor)
