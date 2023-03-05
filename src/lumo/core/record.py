import warnings
from numbers import Number

from . import Attr
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


class Record:
    """Record class for storing and computing metrics over a window of steps.

    Attributes:
        **kwargs: Additional properties to add to the record object.

    Properties:
        stage (str): The stage of the training process, such as 'train' or 'eval'.

    Methods:
        avg(): Computes the average value of the recorded metrics.
        agg(): Computes the aggregated value of the recorded metrics.
        __str__(): Returns a string representation of the recorded metrics.
        tostr(): Alias for __str__().
        record(metric, global_step=None): Records a metric for the current step.
        clear(): Clears the recorded metrics.
        flush(): Clears the cache of recorded metrics.
    """

    def __init__(self, **kwargs):
        self._prop = {}
        self._prop.update(kwargs)
        self._cache = []
        self._agg = OrderedDict()  # type:Dict[str,AggItem]

    def avg(self) -> Attr:
        """DEPRECATED: Computes the average value of the recorded metrics."""
        warnings.warn('avg will be deprecated in later version, use agg() instead.')
        return self.agg()

    def agg(self) -> Attr:
        """Computes the aggregated value of the recorded metrics."""
        res = Attr()
        for k, v in self._agg.items():
            res[k] = v.res
        return res

    @property
    def stage(self):
        """Gets the stage of the training process, such as 'train' or 'eval'."""
        return self._prop['stage']

    def __str__(self):
        """Returns a string representation of the recorded metrics."""
        res = self.agg()
        rep = []
        for k, v in res.items():
            if isinstance(v, float):
                rep.append(f'{k}={v:.4g}')
            else:
                rep.append(f'{k}={str(v)}')

        return ', '.join(rep)

    def tostr(self):
        """Alias for __str__()."""
        return str(self)

    def record(self, metric, global_step=None):
        """Records a metric for the current step."""
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
        """Clears the recorded metrics."""
        self._agg.clear()
        self._cache.clear()

    def flush(self):
        """Clears the cache of recorded metrics."""
        self._cache.clear()


class AggItem:
    """
    A class that aggregates a sequence of values according to a specified strategy.

    Attributes:
        stg (str): A string that specifies the strategy to be used for aggregation.
        _last (int): The last value added to the aggregation.
        acc (int): The accumulated value after aggregation.
        c (int): The count of values added to the aggregation.
    """

    def __init__(self, stg):
        self.stg = stg
        self._last = 0
        self.acc = 0
        self.c = 0

    @property
    def res(self):
        """
        Computes the result of the aggregation.

        Returns:
            int: The result of the aggregation according to the specified strategy.
        """
        if self.stg == 'mean':
            return self.acc / self.c

        if self.stg in {'min', 'max', 'last'}:
            return self.acc

        if self.stg == 'sum':
            return self.acc

        return self.acc

    @property
    def last(self):
        """
        Returns the last value added to the aggregation by `update`.

        Returns:
            int: The last value added to the aggregation.
        """
        return self._last

    def update(self, val):
        """
        Updates the aggregation with a new value.

        Args:
            val (int): The new value to add to the aggregation.
        """
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
