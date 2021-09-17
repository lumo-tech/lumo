"""
Used for recording data
"""
import numpy as np
from collections.abc import ItemsView
from collections import OrderedDict
from numbers import Number
from typing import Union, Iterator, Tuple

import torch

from ..base_classes import attr
from ..base_classes.trickitems import NoneItem
from lumo.utils.fmt import to_ndarray, detach


class Meter:
    def __init__(self):
        self._avg = {}
        self._rec = attr()
        self._stage = 'default'

    def __setattr__(self, key: str, value):
        if key.startswith('_'):
            super(Meter, self).__setattr__(key, value)
        else:
            self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        if item in self._rec:
            return self._rec[item]
        return NoneItem()

    def __setitem__(self, key, value):
        self._rec[key] = value
        if key not in self._avg or self._stage != 'default':
            value = to_ndarray(value)
            dtype = value.dtype.name
            isscalar = value.size == 1
            if self._stage == 'default':
                _avg = 'last'
                if 'float' in dtype:
                    _avg = 'smean'
            else:
                if self._stage in {'min', 'max'} and not isscalar:
                    raise TypeError('Only support min/max operator on scaler values.')
                elif self._stage in {'min', 'max', 'sum', 'mean', 'smean'} and 'str' in dtype:
                    raise TypeError('Only support min/max/sum/mean operator on numbers.')
                _avg = self._stage

            self.set_avg_method(key, _avg)

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

    def update(self, dic: dict):
        for k, v in dic.items():
            self[str(k)] = v
        return self

    def set_avg_method(self, key, typ):
        self._avg[key] = typ

    def items(self):
        return ItemsView(self)

    def keys(self):
        return self._rec.keys()


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


class AvgMeter:

    def __init__(self):
        self._avg = {}
        self._rec = attr()

    def __repr__(self):
        return ' | '.join([f'{k}: {v}' for k, v in self._rec.items()])

    def __setitem__(self, key, value):
        m = Meter()
        m[key] = value
        self.update(m)

    def __setattr__(self, key: str, value):
        if key.startswith('_'):
            super(AvgMeter, self).__setattr__(key, value)
        else:
            self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        if item in self._rec:
            return self._rec[item].cur
        raise KeyError(item)

    __str__ = __repr__

    def __iter__(self):
        yield from self.keys()

    def serialize(self) -> OrderedDict:
        res = OrderedDict()
        for k, v in self.items():
            res[k] = f'{v}'
        return res

    def scalar_items(self) -> Iterator[Tuple[str, Number]]:
        for k, v in self.items():
            if v.isnumber and v.isscalar:
                yield k, to_ndarray(v.res).item()

    def keys(self):
        return self._rec.keys()

    def items(self) -> ItemsView:
        return ItemsView(self)

    def update(self, meter: Union[Meter, dict]):
        if isinstance(meter, dict):
            meter = Meter().update(meter)

        self._avg.update(meter._avg)
        for k, v in meter.items():
            if k in self._rec:
                try:
                    self._rec[k].update(v)
                except TypeError:
                    self._rec[k] = AvgItem(v, self._avg[k])
            else:
                self._rec[k] = AvgItem(v, self._avg[k])

    def clear(self):
        self._rec.clear()

    def record(self, key, value, gb_method=None):
        # meter = Meter()
        # meter[key] = value
        # if gb_method is not None:
        raise NotImplementedError()
