import warnings
from numbers import Number

from . import Attr
from .meter import Meter, ReduceItem
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
        self._agg = OrderedDict()  # type:Dict[str,ReduceItem]

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
        # print(agg)
        # print(meter)
        for k, v in meter.items():
            stg = agg.get(k, 'last')
            item = self._agg.get(k, None)
            if item is None:
                item = ReduceItem(gb_method=stg)
            item.update(v)
            self._agg[k] = item

    def clear(self):
        """Clears the recorded metrics."""
        self._agg.clear()
        self._cache.clear()

    def flush(self):
        """Clears the cache of recorded metrics."""
        self._cache.clear()
