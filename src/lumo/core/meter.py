"""
Used for recording data
"""
from collections import OrderedDict
from collections.abc import ItemsView
from numbers import Number
from typing import Iterator, Tuple, Mapping

import numpy as np
import torch

from lumo.utils.fmt import to_ndarray, detach, is_scalar


class Meter:
    """
    A class for recording and managing metrics.

    Attributes:
        _prop (dict): A dictionary to store properties of the meter.
        _rec (OrderedDict): An ordered dictionary to record the metrics and their values.
        _avg (dict): A dictionary to store the aggregation method for each metric.

    Methods:
        sorted() -> 'Meter': Returns a new meter with the metrics sorted by their names.
        todict() -> OrderedDict: Returns the recorded metrics as an ordered dictionary.
        update(dic: Mapping) -> 'Meter': Updates the meter with the given dictionary of metrics.
        serialize() -> OrderedDict: Returns a dictionary representation of the meter.
        items() -> ItemsView: Returns a view object containing the (metric, value) pairs.
        keys() -> KeysView: Returns a view object containing the metric names.
        scalar_items() -> Iterator[Tuple[str, Number]]: Returns an iterator over the (metric, value) pairs with scalar values.

    Properties:
        sum: Sets the aggregation method to 'sum'.
        mean: Sets the aggregation method to 'mean'.
        last: Sets the aggregation method to 'last'.
        max: Sets the aggregation method to 'max'.
        min: Sets the aggregation method to 'min'.
        smean: Sets the aggregation method to 'smean'.
    """

    def __init__(self):
        self._prop = {}
        self._rec = OrderedDict()
        self._avg = {}

    def sorted(self) -> 'Meter':
        """
         Returns a new meter with the metrics sorted by their names.

         Returns:
             A new meter with the metrics sorted by their names.
         """
        m = Meter()

        m._prop = self._prop
        for k in sorted(self._avg.keys()):
            m._avg[k] = self._avg[k]
            m._rec[k] = self._rec[k]
        return m

    def todict(self):
        """
        Returns the recorded metrics as an ordered dictionary.

        Returns:
            An ordered dictionary containing the recorded metrics and their values.
        """
        return self._rec

    @property
    def _stage(self):
        return self._prop.get('stage', 'default')

    @_stage.setter
    def _stage(self, value):
        self._prop['stage'] = value

    def __setattr__(self, key: str, value):
        """
        Sets the value of an attribute.

        Args:
            key (str): The name of the attribute.
            value: The value to set the attribute to.
        """
        if key.startswith('_'):
            super(Meter, self).__setattr__(key, value)
        else:
            self[key] = value

    def __getattr__(self, item):
        """
        Returns the value of a metric.

        Args:
            item: The name of the metric.

        Returns:
            The value of the metric.
        """
        return self[item]

    def __getitem__(self, item):
        """
        Returns the value of a metric.

        Args:
            item: The name of the metric.

        Returns:
            The value of the metric.
        """
        return self._rec[item]

    def __setitem__(self, key, value):
        """
        Sets the value of a metric.

        Args:
            key: The name of the metric.
            value: The value to set the metric to.
        """
        value = to_ndarray(value)

        stg = self._avg.get(key, None)
        isscalar = value.size == 1

        if stg is None:  # Auto infer a stg method for the value
            dtype = value.dtype.name

            # sanity check
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
            stg = self._stage

        if isscalar:
            value = value.item()
        elif stg == {'min', 'max', 'sum'}:
            value = getattr(np, stg)(value).item()
        elif stg == 'mean':
            value = [getattr(np, stg)(value).item(), value.size]

        self._rec[key] = value
        self._stage = 'default'

    def __repr__(self):
        """
        Returns a string representation of the meter.

        Returns:
            A string representation of the meter.
        """
        return ' | '.join([f'{k}: {v}' for k, v in self._rec.items()])

    def __iter__(self):
        """
        Returns an iterator over the metric names.

        Returns:
            An iterator over the metric names.
        """
        yield from self.keys()

    @property
    def sum(self):
        """
        Sets the aggregation method to 'sum'.

        Returns:
            The meter itself.
        """
        self._stage = 'sum'
        return self

    @property
    def mean(self):
        """
        Sets the aggregation method to 'mean'.

        Returns:
            The meter itself.
        """
        self._stage = 'mean'
        return self

    @property
    def test_mean(self):
        """
        Sets the aggregation method to 'mean'.

        Returns:
            The meter itself.
        """
        self._stage = 'mean'
        return self

    @property
    def last(self):
        """
        Sets the aggregation method to 'last'.

        Returns:
            The meter itself.
        """
        self._stage = 'last'
        return self

    @property
    def max(self):
        """
        Sets the aggregation method to 'max'.

        Returns:
            The meter itself.
        """
        self._stage = 'max'
        return self

    @property
    def min(self):
        """
        Sets the aggregation method to 'min'.

        Returns:
            The meter itself.
        """
        self._stage = 'min'
        return self

    @property
    def smean(self):
        """
        Sets the aggregation method to 'smean'.

        Returns:
            The meter itself.
        """
        self._stage = 'smean'
        return self

    def update(self, dic: Mapping) -> 'Meter':
        """
        Updates the meter with the given dictionary of metrics.

        Args:
            dic (Mapping): A dictionary containing the metrics and their values.

        Returns:
            The meter itself.
        """
        for k, v in dic.items():
            self[str(k)] = v
        return self

    def serialize(self) -> OrderedDict:
        """
        Returns a dictionary representation of the meter.

        Returns:
            An ordered dictionary containing the metrics and their string values.
        """
        res = OrderedDict()
        for k, v in self.items():
            res[k] = f'{v}'
        return res

    def items(self) -> ItemsView:
        """
        Returns a view object containing the (metric, value) pairs.

        Returns:
            A view object containing the (metric, value) pairs.
        """
        return self._rec.items()

    def keys(self):
        """
        Returns a view object containing the metric names.

        Returns:
            A view object containing the metric names.
        """
        return self._rec.keys()

    @staticmethod
    def from_dict(dic: Mapping):
        """
        Returns a new meter with the given dictionary of metrics.

        Args:
            dic (Mapping): A dictionary containing the metrics and their values.

        Returns:
            A new meter with the given metrics and values.
        """
        m = Meter()
        for k, v in dic.items():
            m[k] = v
        return m

    def scalar_items(self) -> Iterator[Tuple[str, Number]]:
        """
        Returns an iterator over the (metric, value) pairs with scalar values.

        Returns:
            An iterator over the (metric, value) pairs with scalar values.
        """
        for k, v in self.items():
            nd = to_ndarray(v)
            if is_scalar(nd):
                yield k, nd.item()


class ReduceItem:
    """Class that reduces a sequence of values to a single value according to a given method.

    Attributes:
        SLIDE_WINDOW_SIZE (int): The size of the sliding window used for averaging (default: 100).
        EXP_WEIGHT (float): The exponential weight used for computing the sliding window offset (default: 0.75).

    Args:
        item (optional): The initial value (default: None).
        gb_method (optional): The reduction method (default: None). Can be one of {'slide', 'mean', 'sum', 'max', 'min', 'last'}.

    Raises:
        AssertionError: If the reduction method is 'min' or 'max' and the input is not a scalar.

    Methods:
        __repr__(): Returns a string representation of the current reduction value.
        __str__(): Returns the same string representation as __repr__().
        update(item): Updates the reduction value with a new item.
        res: Returns the current reduction value.

    Examples:
        >>> r = ReduceItem(gb_method='mean')
        >>> r.update(2)
        >>> r.update(3)
        >>> r.res
        2.5
    """
    SLIDE_WINDOW_SIZE = 100
    EXP_WEIGHT = 0.75

    def __init__(self, item=None, gb_method=None):
        """
        Initializes a new ReduceItem instance.

        Args:
            item (optional): The initial value (default: None).
            gb_method (optional): The reduction method (default: None). Can be one of {'slide', 'mean', 'sum', 'max', 'min', 'last'}.

        Raises:
            AssertionError: If the reduction method is 'min' or 'max' and the input is not a scalar.
        """
        self.gb_method = gb_method  # groupby method
        self.acc = []
        if item is not None:
            self.acc.append(detach(item))
        self.c = len(self.acc)
        self.cur = item
        if gb_method == 'max':
            self._last = -1e12
        elif gb_method == 'min':
            self._last = 1e12
        else:
            self._last = 0

        self._res = self.last

        self.offset = 0
        self.nd = to_ndarray(item)
        self.isint = 'int' in self.nd.dtype.name
        self.isnumber = (self.isint or 'float' in self.nd.dtype.name) and isinstance(item, (np.ndarray,
                                                                                            torch.Tensor))
        self.isscalar = self.nd.size == 1
        if not self.isscalar and gb_method in {'min', 'max'}:
            raise AssertionError(f'{gb_method} method only support scaler')

    def __repr__(self):
        """
        Returns a string representation of the current reduction value.

        Returns:
            str: The string representation of the current reduction value.
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
        """Updates the reduction value with a new item."""
        self.cur = item
        count = 1
        if isinstance(item, list) and len(item) == 2:
            item, count = item

        item = detach(item)

        avg = self.gb_method
        if self.isnumber:
            self.offset = self.offset * ReduceItem.EXP_WEIGHT + abs(item - self.last) * (1 - ReduceItem.EXP_WEIGHT)

        if avg == 'slide':
            self.acc.append(item)
            if len(self.acc) > ReduceItem.SLIDE_WINDOW_SIZE:
                self.acc.pop(0)
            self._last = self.cur
        elif avg in {'mean', 'sum'}:
            if len(self.acc) == 0:
                self.acc.append(0)
            self.acc[0] = self.acc[0] + item * count
            self.c += count
        elif avg == 'max':
            self._res = max(self.cur, self._res)
        elif avg == 'min':
            self._res = min(self.cur, self._res)

        self._last = item

    @property
    def last(self):
        return self._last

    @property
    def res(self):
        """
        Returns the current reduction value.

        Returns:
            float: The current reduction value.
        """
        avg = self.gb_method
        if avg == 'slide':
            return np.mean(self.acc)
        if avg == 'mean':
            return self.acc[0] / self.c
        elif avg == 'sum':
            return self.acc[0]
        elif avg in {'max', 'min'}:
            return self._res
        elif avg == 'last':
            return self.last
        return self.cur
