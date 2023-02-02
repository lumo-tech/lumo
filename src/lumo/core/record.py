import warnings
from numbers import Number

from . import Attr
from .metaclasses import PropVar
from .meter import Meter
import torch
import numpy as np
from typing import NewType, Mapping, Union, Sequence, Dict
from collections import OrderedDict

MetricType = NewType('MetricType', Union[Mapping, Meter, Sequence,
                                         Number, torch.Tensor, np.ndarray
])

from collections import namedtuple

record_event = namedtuple('record_event', ['global_step', 'metric', 'time'])


def wrap_result(metric: MetricType) -> Meter:
    """
    Wrap any form of metric data into Meter form.

    """
    if isinstance(metric, (Meter,)):
        return metric
    elif isinstance(metric, Mapping):
        return Meter.from_dict(metric)
    elif isinstance(metric, Sequence):
        return Meter.from_dict({f'm{i}': v for i, v in enumerate(metric)})
    elif isinstance(metric, (torch.Tensor, np.ndarray, float, int)):
        return Meter.from_dict({'metric': metric})
    return Meter()


class Record(metaclass=PropVar):
    def __init__(self, window_size=500, **kwargs):
        self._prop.update(kwargs)
        self._cache = []
        self._agg = OrderedDict()  # type:Dict[str,AggItem]

    def avg(self) -> Attr:
        warnings.warn('avg will be deprecated in later version, use agg() instead.')
        return self.agg()

    def agg(self) -> Attr:
        res = Attr()
        for k, v in self._agg.items():
            res[k] = v.res
        return res

    @property
    def stage(self):
        return self._prop['stage']

    def __str__(self):
        res = self.agg()
        rep = []
        for k, v in res.items():
            if isinstance(v, float):
                rep.append(f'{k}={v:.4g}')
            else:
                rep.append(f'{k}={str(v)}')

        return ', '.join(rep)

    def tostr(self):
        return str(self)

    def record(self, metric, global_step=None):
        meter = wrap_result(metric)
        agg = meter._avg

        for k, v in meter.items():
            stg = agg.get(k, 'last')
            item = self._agg.get(k, None)
            if item is None:
                item = AggItem(stg)
            item.update(v)
            self._agg[k] = item

    def clear(self):
        self._agg.clear()
        self._cache.clear()

    def flush(self):
        self._cache.clear()


class AggItem:
    def __init__(self, stg):
        self.stg = stg
        self._last = 0
        self.acc = 0
        self.c = 0

    @property
    def res(self):
        if self.stg == 'mean':
            return self.acc / self.c

        if self.stg in {'min', 'max', 'last'}:
            return self.acc

        if self.stg == 'sum':
            return self.acc

        return self.acc

    @property
    def last(self):
        return self._last

    def update(self, val):
        if self.stg == 'last':
            self.acc = val
        elif self.stg == 'min':
            self.acc = min(self.acc, val)
        elif self.stg == 'max':
            self.acc = max(self.acc, val)
        elif self.stg in {'mean', 'sum'}:
            self.acc += val
            self.c += 1
        self._last = val
