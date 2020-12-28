"""

"""
import warnings
import copy
import torch
import numpy as np
from collections import OrderedDict

from typing import Any, Iterable
from thexp.base_classes.trickitems import _ContainWrap
from thexp.base_classes.metaclasses import meta_attr

_attr_clss = {}


class attr(OrderedDict, metaclass=meta_attr):
    """
    An ordered defaultdict, the default class is attr itself.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        self = super().__new__(cls, *args, **kwargs)
        # self._class_name = cls.__name__
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
        if isinstance(k, _ContainWrap):
            k = k.value

        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            if isinstance(cwk, _ContainWrap):
                return super().__getitem__(ks[0])

            try:
                return super().__getitem__(ks[0])
            except:
                self[ks[0]] = attr()
                return self[ks[0]]

        cur = self
        for tk in ks:
            if isinstance(cwk, _ContainWrap):
                cur = cur.__getitem__(_ContainWrap(tk))
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
                cur = cur.__getattr__(tk)
            cur[ks[-1]] = v

    def __contains__(self, o: object) -> bool:
        try:
            _ = self[_ContainWrap(o)]
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

    def items(self, toggle=False):
        if toggle:
            yield '_class_name', self.__class__.__name__
        for k, v in super().items():
            yield k, v

    def raw_items(self):
        return self.items(toggle=True)

    def jsonify(self) -> dict:
        """
        获取可被json化的dict，目前仅支持 数字类型、字符串、bool、list/set 类型
        :return:
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
                            nv.append(_nattr().jsonify())
                        else:  # set,list
                            _tmp = attr()
                            _tmp['tmp'] = vv
                            nv.append(_tmp.jsonify()['tmp'])
                res[k] = nv
            elif v is None:
                res[k] = _nattr().jsonify()

        return res

    def hash(self) -> str:
        from ..utils.paths import hash
        return hash(self)

    def copy(self):
        return self.__copy__()

    def replace(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v
        return self

    @classmethod
    def from_dict(cls, dic: dict):
        # res = cls()
        cls_name = dic.get('_class_name', cls.__name__)
        if cls_name not in _attr_clss:
            res = attr()
            from .errors import AttrTypeNotFoundWarning
            warnings.warn('{} not found, will use class attr to receive values.'.format(cls_name),
                          AttrTypeNotFoundWarning)
        else:
            res = _attr_clss[cls_name].__new__(_attr_clss[cls_name])

        if '_class_name' in dic:
            dic.pop('_class_name')

        if isinstance(res, _nattr):
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


class _nattr(attr):
    pass


class _tpttr(attr):  # pytorch/numpy wrapper
    def __init__(self, arr):
        super().__init__()
        self.arr = arr.tolist()
        self.torch = isinstance(arr, torch.Tensor)
