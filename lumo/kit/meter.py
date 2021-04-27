"""
Used for recording data
"""
from collections import OrderedDict
from numbers import Number
from typing import Any, Union

import numpy as np
import torch

from ..base_classes.errors import MeterNameException
from ..base_classes.trickitems import NoneItem


class _Format:
    def __call__(self, value):
        raise NotImplementedError()


class _IntFmt(_Format):
    def __call__(self, value):
        return f"{value:.0f}"


class _FloatFmt(_Format):
    def __init__(self, precision=2):
        self.prec = precision

    def __call__(self, value):
        return f"{value:.{self.prec}f}"


class _PercentFmt(_FloatFmt):
    def __call__(self, value):
        return f"{value:.{self.prec}%}"


class _TensorFmt(_FloatFmt):

    def __call__(self, value):
        if len(value.shape) != 0:
            value = value.item()
        return f"{value:.{self.prec}f}"


class _InlineStrFmt(_Format):

    def __call__(self, value):
        l = str(value).split("\n")
        if len(l) > 1:
            return f"{l[0]}..."
        else:
            return l[0]


class _AvgItem:
    """
    用于保存累积均值的类
    avg = AvgItem()
    avg += 1 # avg.update(1)
    avg += 2
    avg += 3

    avg.item = 3 #(last item)
    avg.avg = 2 #(average item)
    avg.sum = 6
    """

    def __init__(self, weight_decay=0.01, precision=4) -> None:
        super().__init__()
        self._sum = None
        self._count = 0
        self._last = None
        self._w = weight_decay
        self._p = precision

    def update(self, other):
        if self._sum is None:
            self._sum = other
        else:
            self._sum = self._sum + other
        self._count += 1
        self._last = other

    @property
    def last(self) -> Number:
        if self._last is None:
            return 0
        return self._last

    @property
    def avg(self) -> Number:
        if self._sum is None:
            return 0
        return self._sum / self._count

    def __repr__(self) -> str:
        return f"{self.avg:.{self._p}f}"

    def __format__(self, format_spec):
        return f"{self.avg}:{format_spec}"


class Meter:
    """

    Examples:
    --------
    m = Meter()
    m.short()
    m.int()
    m.float()
    m.percent()

    m.k += 10
    m.k.update()

    m.k.backstep()
    """

    def __init__(self, precision=4):
        self._prec = precision
        self._meter_dict = OrderedDict()
        self._format_dict = dict()
        self._convert_type = []

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name.endswith("_"):
            raise MeterNameException(f"Meter name can't end with '_', but got '{name}'")
        else:
            value = self._convert(value)

            if isinstance(value, int) and name not in self._format_dict:
                self.int(name)
            elif isinstance(value, float) and name not in self._format_dict:
                self.float(name)
            elif isinstance(value, torch.Tensor) and name not in self._format_dict:
                if len(value.shape) == 0 or (sum(value.shape) == 1):
                    self.tensorfloat(name)
                else:
                    self.str_in_line(name)

            self._meter_dict[name] = value

    def __getattr__(self, item):
        if item.endswith("_"):
            return item.rstrip("_")

        if item not in self._meter_dict:
            return NoneItem()
        else:
            return self._meter_dict[item]

    def __getitem__(self, item):
        item = str(item)
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __iter__(self):
        return iter(self._meter_dict)

    def __repr__(self):
        log_dict = self.serialize()
        return " | ".join(["{}: {}".format(k, v) for k, v in log_dict.items()])

    def _convert(self, val):
        if type(val) in {int, float, bool, str}:
            return val
        for tp, func in self._convert_type:
            if type(val) == tp:
                val = func(val)
        return val

    def serialize(self) -> OrderedDict:
        """format all value to be strings to make sure it can be printed without Exception"""
        log_dict = OrderedDict()
        for k, v in self._meter_dict.items():
            if k in self._format_dict:
                v = self._format_dict[k](v)
            log_dict[k] = v
        return log_dict

    def int(self, key: str):
        """
        Variable with name `key` will be formatted as int when log.
        Args:
            key: Variable name
        """
        self._format_dict[key] = _IntFmt()
        return self

    def float(self, key: str, precision=3):
        """
        Variable with name `key` will be formatted as float when log.

        Args:
            key: Variable name
            precision:
        """
        self._format_dict[key] = _FloatFmt(precision)
        return self

    def percent(self, key: str, precision=2):
        """
        Variable with name `key` will be formatted as a percentage when log.
        Args:
            key: Variable name
            precision:
        """
        self._format_dict[key] = _PercentFmt(precision)
        return self

    def tensorfloat(self, key: str, precision=4):
        """
        Format torch.Tensor object as a float when log.
        Args:
            key: Variable name.
            precision:
        """
        self._format_dict[key] = _TensorFmt(precision)
        return self

    def str_in_line(self, key: str):
        self._format_dict[key] = _InlineStrFmt()
        return self

    def add_format_type(self, type, func):
        self._convert_type.append((type, func))
        return self

    def items(self):
        """like dict.items()"""
        for k, v in self._meter_dict.items():
            yield k, v

    def array_items(self):
        """
        return items that is a Number/torch.Tensor/np.ndarray object
        Returns:
            iterated (k, v) sequence, where v is an instance of Number/torch.Tensor/np.ndarray
        """
        for k, v in self._meter_dict.items():
            if isinstance(v, (Number, torch.Tensor, np.ndarray)):
                yield k, v

    def numeral_items(self):
        """
        return items that can be converted into one scalar.
        Returns:
            iterated (k, v) sequence, where v is int or float object
        Notes:
            all array(torch.Tensor/np.ndarray) that only has one element will be yiled, either.
        """
        for k, v in self._meter_dict.items():
            if isinstance(v, (int, float)):
                yield k, v
            elif isinstance(v, torch.Tensor):
                try:
                    yield k, v.detach().cpu().item()
                except ValueError:  # not an one element tensor
                    continue
            elif isinstance(v, np.ndarray):
                try:
                    yield k, v.item()
                except ValueError:  # not an one element tensor
                    continue

    def update(self, meter: dict):
        """like dict.update()"""
        for k, v in meter.items():
            self[k] = v
        return self


class AvgMeter(Meter):
    """

    """

    def __init__(self, weight_decay=0.75, precision=4):
        super().__init__(precision)
        self._weight_decay = weight_decay

    def __getitem__(self, item) -> _AvgItem:
        return super().__getitem__(item)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name.endswith("_"):
            raise MeterNameException(f"Meter name can't end with '_', but got '{name}'")
        else:
            value = self._convert(value)
            if isinstance(value, int) and name not in self._format_dict:
                self.int(name)
            elif isinstance(value, float) and name not in self._format_dict:
                self.float(name)
            elif isinstance(value, torch.Tensor) and name not in self._format_dict:
                if len(value.shape) == 0 or (sum(value.shape) == 1):
                    self.tensorfloat(name)
                else:
                    self.str_in_line(name)

            if isinstance(value, (torch.Tensor, np.ndarray)):
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                if len(value.shape) == 0 or sum(value.shape) == 1:
                    if name not in self._meter_dict:
                        self._meter_dict[name] = self.avg_item()
                    self._meter_dict[name].update(value)
                else:
                    self._meter_dict[name] = value
            elif name in self._meter_dict and isinstance(self._meter_dict[name], _AvgItem):
                self._meter_dict[name].update(value)
            elif isinstance(value, (float)):
                if name not in self._meter_dict:
                    self._meter_dict[name] = self.avg_item()
                self._meter_dict[name].update(value)
            else:
                self._meter_dict[name] = value

    def avg_item(self):
        return _AvgItem(weight_decay=0.75, precision=self._prec)

    def update(self, meter: Union[Meter, dict]):
        if meter is None:
            return
        for k, v in meter.items():
            self[k] = v
        if isinstance(meter, Meter):
            self._format_dict.update(meter._format_dict)
            self._convert_type.extend(meter._convert_type)

    def serialize(self):
        log_dict = OrderedDict()
        for k, v in self._meter_dict.items():
            if k in self._format_dict:
                if isinstance(v, _AvgItem):
                    v = v.avg
                v = self._format_dict[k](v)
            log_dict[k] = v
        return log_dict

    def array_items(self):
        for k, v in self._meter_dict.items():
            if isinstance(v, (int, float, torch.Tensor, np.ndarray)):
                yield k, v
            elif isinstance(v, _AvgItem):
                yield k, v.avg

    def numeral_items(self):
        """"""
        for k, v in self._meter_dict.items():
            if isinstance(v, (int, float)):
                yield k, v
            elif isinstance(v, torch.Tensor):
                try:
                    yield k, v.detach().cpu().item()
                except:
                    continue
            elif isinstance(v, np.ndarray):
                try:
                    yield k, v.item()
                except:
                    continue
            elif isinstance(v, _AvgItem):
                yield k, v.avg

    @property
    def meter(self):
        """try to get newest value"""
        return _InMeter(self)

    @property
    def avg(self):
        """try to get average value"""
        return _InMeter(self, mode='avg')


class _InMeter():
    def __init__(self, avgmeter: AvgMeter, mode='newest'):
        self._avg = avgmeter
        self._mode = mode

    def __getattr__(self, item):
        if item not in self._avg._meter_dict:
            raise AttributeError(item)
        res = self._avg._meter_dict[item]
        if isinstance(res, _AvgItem):
            if self._mode == 'newest':
                res = res.last
            else:
                res = res.avg
        return res

    def __getitem__(self, item: str):
        return self.__getattr__(item)
