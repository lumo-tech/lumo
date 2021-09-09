"""

"""
import copy
import warnings
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import torch
from joblib import hash

from lumo.base_classes.metaclasses import meta_attr

# from lumo.utils.keys import
_attr_clss = {}

ATTR_TYPE = '_type'


class attr(OrderedDict, metaclass=meta_attr):
    """
    An ordered defaultdict, the default class is attr itself.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        self = super().__new__(cls, *args, **kwargs)
        return self

    @staticmethod
    def __parse_value(v):
        if isinstance(v, attr):
            pass
        elif isinstance(v, dict):
            v = attr.from_dict(v)
        return v

    def __getattr__(self, item):
        if item not in self:
            self[item] = attr()
        return self[item]

    def __setattr__(self, name: str, value: Any) -> None:
        value = self.__parse_value(value)
        self[name] = value

    def __getitem__(self, k):
        cwk = k
        if isinstance(k, _contains):
            k = k.value

        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            if isinstance(cwk, _contains):
                return super().__getitem__(ks[0])

            try:
                return super().__getitem__(ks[0])
            except KeyError:
                self[ks[0]] = attr()
                return self[ks[0]]

        cur = self
        for tk in ks:
            if isinstance(cwk, _contains):
                cur = cur.__getitem__(_contains(tk))
            else:
                cur = cur.__getitem__(tk)
        return cur

    def __setitem__(self, k, v):
        v = self.__parse_value(v)

        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            super().__setitem__(ks[0], v)
        else:
            cur = self
            for tk in ks[:-1]:
                ncur = cur.__getattr__(tk)
                if not isinstance(ncur, attr):
                    cur[tk] = attr()
                    ncur = cur[tk]
                cur = ncur

            cur[ks[-1]] = v

    def __contains__(self, o: object) -> bool:
        try:
            _ = self[_contains(o)]
            return True
        except:
            return False

    def __copy__(self):
        # res = attr()
        return self.from_dict(self)

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def walk(self):
        for k, v in self.items():
            if isinstance(v, attr):
                for ik, iv in v.walk():
                    ok = "{}.{}".format(k, ik)
                    yield ok, iv
            else:
                yield k, v

    def __eq__(self, o: object) -> bool:
        if isinstance(o, dict):
            return self.hash() == hash(o)
        return False

    def items(self, toggle=False):
        if toggle:
            yield ATTR_TYPE, self.__class__.__name__
        for k, v in super().items():
            yield k, v

    def raw_items(self):
        return self.items(toggle=True)

    def pickify(self) -> dict:
        """
        Return a serialized dict which can be dumped by pickle.
        You can deserialize this dict by `from_dict()`
        """
        res = dict()
        for k, v in self.raw_items():
            if isinstance(v, attr):
                v = v.pickify()
                res[k] = v
            elif isinstance(v, (Iterable)):
                if not isinstance(v, (torch.Tensor, np.ndarray, str)):
                    nv = []
                    for vv in v:
                        if isinstance(vv, dict):
                            if isinstance(vv, attr):
                                nv.append(vv.pickify())
                            else:
                                nv.append(attr.from_dict(vv).pickify())
                        elif vv is None:
                            nv.append(_none().pickify())
                        else:  # set,list
                            # _tmp = attr()
                            # _tmp['tmp'] = vv
                            nv.append(vv)
                else:
                    nv = v
                res[k] = nv
            elif v is None:
                res[k] = _none().pickify()
            else:
                res[k] = v

        return res

    def jsonify(self) -> dict:
        """
        Return a serialized dict which can be dumped in json format.
        You can deserialize this dict by `from_dict()`
        """
        import numbers
        res = dict()
        for k, v in self.raw_items():
            if isinstance(v, (numbers.Number, str)):
                res[k] = v
            elif isinstance(v, attr):
                v = v.jsonify()
                res[k] = v
            elif isinstance(v, (Iterable)):
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    nv = _tpttr(v).jsonify()
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
        return hash(self)

    def copy(self):
        """deep copy"""
        return self.__copy__()

    def replace(self, **kwargs):
        """An alias of update()"""
        for k, v in kwargs.items():
            self[k] = v
        return self

    @classmethod
    def from_dict(cls, dic: dict):
        # res = cls()
        cls_name = dic.get(ATTR_TYPE, cls.__name__)
        if cls_name not in _attr_clss:
            res = attr()
            from .errors import AttrTypeNotFoundWarning
            warnings.warn('{} not found, will use class attr to receive values.'.format(cls_name),
                          AttrTypeNotFoundWarning)
        else:
            res = _attr_clss[cls_name].__new__(_attr_clss[cls_name])

        if ATTR_TYPE in dic:
            dic.pop(ATTR_TYPE)

        if isinstance(res, _none):
            return None
        elif isinstance(res, _tpttr):
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


class _tpttr(attr):
    """
    A tricky item used in attr to parse pytorch and numpy array.
    """

    def __init__(self, arr):
        super().__init__()
        self.arr = arr.tolist()
        self.torch = isinstance(arr, torch.Tensor)


class _contains():
    """
    A tricky item that used in attr to discriminate the difference between `get()` and `contains()`
    """

    def __init__(self, value):
        self.value = value
