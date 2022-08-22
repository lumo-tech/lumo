"""
Used for recording data
"""
from collections import OrderedDict
from collections.abc import ItemsView
from numbers import Number
from typing import Union, Iterator, Tuple, Mapping, Sequence

import numpy as np
import torch

from lumo.utils.fmt import to_ndarray, detach, is_scalar
from lumo.core import PropVar


class Meter(metaclass=PropVar):
    def __init__(self):
        self._rec = {}
        self._avg = {}

    def sorted(self) -> 'Meter':
        m = Meter()
        m._rec = self._rec
        m._avg = OrderedDict()
        m._prop = self._prop
        for k in sorted(self._avg.keys()):
            m._avg[k] = self._avg[k]
        return m

    def todict(self):
        return self._rec

    @property
    def _stage(self):
        return self._prop.get('stage', 'default')

    @_stage.setter
    def _stage(self, value):
        self._prop['stage'] = value

    def __setattr__(self, key: str, value):
        if key.startswith('_'):
            super(Meter, self).__setattr__(key, value)
        else:
            self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        return self._rec[item]

    def __setitem__(self, key, value):
        value = to_ndarray(value)

        stg = self._avg.get(key, None)
        isscalar = value.size == 1

        if stg is None:
            dtype = value.dtype.name

            if self._stage in {'min', 'max'} and not isscalar:
                raise ValueError(
                    f'Only support min/max(a) operator on scalar metrics, but got data of shape {value.shape}.')
            elif self._stage in {'min', 'max', 'sum', 'mean', 'smean'} and 'str' in dtype:
                raise ValueError(f'Only support min/max/sum/mean operator on tensor metrics, but got type {dtype}.')

            if self._stage == 'default':
                if isscalar:
                    if 'int' in dtype:
                        self._stage = 'last'
                    else:
                        self._stage = 'mean'
                else:
                    self._stage = 'last'

            self._avg[key] = self._stage

        if isscalar:
            value = value.item()

        self._rec[key] = value
        self._stage = 'default'

    def __repr__(self):
        return ' | '.join([f'{k}: {v}' for k, v in self._rec.items()])

    def __iter__(self):
        yield from self.keys()

    @property
    def sum(self):
        self._stage = 'sum'
        return self

    @property
    def mean(self):
        self._stage = 'mean'
        return self

    @property
    def last(self):
        self._stage = 'last'
        return self

    @property
    def max(self):
        self._stage = 'max'
        return self

    @property
    def min(self):
        self._stage = 'min'
        return self

    @property
    def smean(self):
        self._stage = 'smean'
        return self

    def update(self, dic: Mapping) -> 'Meter':
        for k, v in dic.items():
            self[str(k)] = v
        return self

    def serialize(self) -> OrderedDict:
        res = OrderedDict()
        for k, v in self.items():
            res[k] = f'{v}'
        return res

    def items(self) -> ItemsView:
        return self._rec.items()

    def keys(self):
        return self._rec.keys()

    @staticmethod
    def from_dict(dic: Mapping):
        m = Meter()
        for k, v in dic.items():
            m[k] = v
        return m

    def scalar_items(self) -> Iterator[Tuple[str, Number]]:
        for k, v in self.items():
            nd = to_ndarray(v)
            if is_scalar(nd):
                yield k, nd.item()


class AvgItem:
    SLIDE_WINDOW_SIZE = 100
    EXP_WEIGHT = 0.75

    def __init__(self, item, gb_method):
        item_ = detach(item)
        self.gb_method = gb_method  # groupby method
        self.acc = [item_]
        self.c = 1
        self.cur = item
        self.last = item_
        self.offset = item_
        self.nd = to_ndarray(item)
        self.isint = 'int' in self.nd.dtype.name
        self.isnumber = (self.isint or 'float' in self.nd.dtype.name) and isinstance(item, (np.ndarray,
                                                                                            torch.Tensor))
        self.isscalar = self.nd.size == 1
        if not self.isscalar and gb_method in {'min', 'max'}:
            raise AssertionError(f'{gb_method} method only support scaler')

    def __repr__(self):
        """
        simpler but more time-comsuming method could be some math function, not in if-else branch, like
            prec =  max(min(8, int(np.ceil(np.log10((1 / (self.offset + 1e-10)))))), 1)
            fmt_str = f'{{:.{prec}f}}'
            return fmt_str.format(res)
        """
        res = self.res
        if self.isscalar:
            res = to_ndarray(res).item()
            if self.isint:
                return f"{res}"
            elif self.isnumber:
                # return f'{res:.4f}'
                if self.offset < 1e-8:
                    return f'{res:.10f}'
                elif self.offset < 1e-6:
                    return f'{res:.8f}'
                elif self.offset < 1e-4:
                    return f'{res:.6f}'
                return f'{res:.4f}'
            else:
                return f'{res}'
        else:
            return f'{res}'

    __str__ = __repr__

    def update(self, item):
        self.cur = item
        item = detach(item)

        avg = self.gb_method
        if self.isnumber:
            self.offset = self.offset * AvgItem.EXP_WEIGHT + abs(item - self.last) * (1 - AvgItem.EXP_WEIGHT)

        if avg == 'slide':
            self.acc.append(item)
            if len(self.acc) > AvgItem.SLIDE_WINDOW_SIZE:
                self.acc.pop(0)
            self.last = self.cur
        elif avg in {'mean', 'sum'}:
            self.acc[0] = self.acc[0] + item
            self.c += 1
        elif avg == 'max':
            self.last = max(self.cur, self.last)
        elif avg == 'min':
            self.last = min(self.cur, self.last)
        elif avg == 'last':
            self.last = item

    @property
    def res(self):
        avg = self.gb_method
        if avg == 'slide':
            return np.mean(self.acc)
        if avg == 'mean':
            return self.acc[0] / self.c
        elif avg == 'sum':
            return self.acc[0]
        elif avg in {'max', 'min', 'last'}:
            return self.last
        return self.cur
