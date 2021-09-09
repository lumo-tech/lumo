"""
Used for recording data
"""
from collections.abc import ItemsView
from collections import OrderedDict
from numbers import Number
from typing import Union, Iterator, Tuple

from ..base_classes import attr
from ..base_classes.trickitems import NoneItem
from lumo.utils.fmt import to_ndarray


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
            isscaler = value.size == 1
            if self._stage == 'default':
                _avg = 'last'
                if 'float' in dtype:
                    _avg = 'mean'
            else:
                if self._stage in {'min', 'max'} and not isscaler:
                    raise TypeError('Only support min/max operator on scaler values.')
                elif self._stage in {'min', 'max', 'sum', 'mean'} and 'str' in dtype:
                    raise TypeError('Only support min/max/sum/mean operator on numbers.')
                _avg = self._stage

            self.add_avg_method(key, _avg)

        self._stage = 'default'

    def __repr__(self):
        return ' | '.join([f'{k}: {v}' for k, v in self._rec.items()])

    def __iter__(self):
        return self.keys()

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

    def update(self, dic: dict):
        for k, v in dic.items():
            self[str(k)] = v
        return self

    def add_avg_method(self, key, typ):
        self._avg[key] = typ

    def add_fmt_method(self, key, fmt):
        self._fmt[key] = fmt

    def items(self):
        return ItemsView(self)

    def keys(self):
        return self._rec.keys()


class AvgItem:
    def __init__(self, item, avg):
        self.avg = avg
        self.acc = item
        self.cur = item
        self.last = item
        self.c = 1
        self.offset = -1
        self.nd = to_ndarray(item)
        self.isint = 'int' in self.nd.dtype.name
        self.isnumber = self.isint or 'float' in self.nd.dtype.name
        self.isscaler = self.nd.size == 1

    def __repr__(self):
        res = self.res
        if self.isscaler:
            res = to_ndarray(res).item()
            if self.isint:
                return f"{res}"
            elif self.isnumber:  #
                if self.offset > 1:
                    return f'{res:.2f}'
                elif self.offset < 1e-2:
                    return f'{res:.4f}'
                elif self.offset < 1e-4:
                    return f'{res:.6f}'
                elif self.offset < 1e-6:
                    return f'{res:.8f}'
                return f'{res:.4f}'
            else:
                return f'{res}'
        else:
            return f'{res}'

    def __str__(self):
        return self.__repr__()

    def update(self, item):
        self.c += 1
        avg = self.avg
        self.cur = item

        if self.isnumber:
            self.offset = abs(self.cur - self.last)

        if avg in {'mean', 'sum'}:
            self.acc += item
        elif avg == 'max':
            self.last = max(self.cur, self.last)
        elif avg == 'min':
            self.last = min(self.cur, self.last)
        elif avg == 'last':
            self.last = item

    @property
    def res(self):
        avg = self.avg
        if avg == 'mean':
            return self.acc / self.c
        elif avg == 'sum':
            return self.acc
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

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        yield from self.keys()

    def serialize(self) -> OrderedDict:
        res = OrderedDict()
        for k, v in self.items():
            res[k] = f'{v}'
        return res

    def scaler_items(self) -> Iterator[Tuple[str, Number]]:
        for k, v in self.items():
            if v.isnumber and v.isscaler:
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
                except:
                    self._rec[k] = AvgItem(v, self._avg[k])
            else:
                self._rec[k] = AvgItem(v, self._avg[k])
