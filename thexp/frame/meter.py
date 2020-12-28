"""
Used for recording data
"""
from collections import OrderedDict
from numbers import Number
from typing import Any, Iterable

import numpy as np
import torch

from ..base_classes.trickitems import AvgItem, NoneItem


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

    def __init__(self):
        self._meter_dict = OrderedDict()
        self._format_dict = dict()
        self._convert_type = []

    def int(self, item: str):
        self._format_dict[item] = lambda x: "{:.0f}".format(x)

    def float(self, key: str, acc=4):
        """
        设置浮点数精度，默认为小数点后四位
        Args:
            key:
            acc:

        Returns:

        """
        self._format_dict[key] = lambda x: "{{:.{}f}}".format(acc).format(x)

    def percent(self, key: str, acc=2):
        self._format_dict[key] = lambda x: "{{:.{}%}}".format(acc).format(x)

    def tensorfloat(self, key: str, acc=4):
        def func(x: torch.Tensor):
            if len(x.shape) == 0:
                return "{{:.{}f}}".format(acc).format(x)
            else:
                return "{{:.{}f}}".format(acc).format(x.item())

        self._format_dict[key] = func

    def str_in_line(self, key: str):
        def func(x):
            l = str(x).split("\n")
            if len(l) > 1:
                return "{}...".format(l[0])
            else:
                return l[0]

        self._format_dict[key] = func

    def add_format_type(self, type, func):
        self._convert_type.append((type, func))

    def _convert(self, val):
        if type(val) in {int, float, bool, str}:
            return val
        # elif isinstance(val, torch.Tensor):
        #     if len(val.shape) == 1 and val.shape[0] == 1:
        #         return val[0]
        for tp, func in self._convert_type:
            if type(val) == tp:
                val = func(val)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name.endswith("_"):
            assert False
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

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __getattr__(self, item):
        if item.endswith("_"):
            return item.rstrip("_")

        if item not in self._meter_dict:
            return NoneItem()
        else:
            return self._meter_dict[item]

    def items(self):
        for k, v in self._meter_dict.items():
            yield k, v

    def array_items(self):
        """任何数字类型的对象"""
        for k, v in self._meter_dict.items():
            if isinstance(v, (Number, torch.Tensor, np.ndarray)):
                yield k, v

    def numeral_items(self):
        """纯可被记录的数字"""
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

    def __iter__(self):
        return iter(self._meter_dict)

    def serialize(self):
        """格式化字典"""
        log_dict = OrderedDict()
        for k, v in self._meter_dict.items():
            if k in self._format_dict:
                v = self._format_dict[k](v)
            log_dict[k] = v
        return log_dict

    def __getitem__(self, item):
        item = str(item)
        return self.__getattr__(item)

    def __repr__(self):
        log_dict = self.serialize()

        return " | ".join(["{}: {}".format(k, v) for k, v in log_dict.items()])

    def update(self, meter):
        for k, v in meter.items():
            self[k] = v
        return self

    def map_update(self, items: Iterable, names: Iterable):
        items, names = list(items), list(names)
        assert len(items) == len(names), 'length of items and names not match'
        for item, name in zip(items, names):
            self[name] = item
        return self


class AvgMeter(Meter):
    """

    """

    def __init__(self):
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name.endswith("_"):
            assert False
        else:
            value = self._convert(value)
            if isinstance(value, int) and name not in self._format_dict:
                self.int(name)
            elif isinstance(value, float) and name not in self._format_dict:
                self.float(name)

            if name in self._meter_dict and isinstance(self._meter_dict[name], AvgItem):
                self._meter_dict[name].update(value)
            elif isinstance(value, (float)):
                if name not in self._meter_dict:
                    self._meter_dict[name] = AvgItem()
                self._meter_dict[name].update(value)
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                if len(value.shape) == 0 or sum(value.shape) == 1:
                    if name not in self._meter_dict:
                        self._meter_dict[name] = AvgItem()
                    self._meter_dict[name].update(value)
                else:
                    self._meter_dict[name] = value
            else:
                self._meter_dict[name] = value

    def __getitem__(self, item) -> AvgItem:
        return super().__getitem__(item)

    # def __delitem__(self, key):

    def update(self, meter):
        if meter is None:
            return
        for k, v in meter._meter_dict.items():
            self[k] = v
        self._format_dict.update(meter._format_dict)
        self._convert_type.extend(meter._convert_type)

    def average(self, k):
        self[k] = AvgItem()

    def serialize(self):
        log_dict = OrderedDict()
        for k, v in self._meter_dict.items():
            if k in self._format_dict:
                v = self._format_dict[k](v)
            log_dict[k] = v
        return log_dict

    def array_items(self):
        for k, v in self._meter_dict.items():
            if isinstance(v, (int, float, torch.Tensor, np.ndarray)):
                yield k, v
            elif isinstance(v, AvgItem):
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
            elif isinstance(v, AvgItem):
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
        if isinstance(res, AvgItem):
            if self._mode == 'newest':
                res = res._item
            else:
                res = res.avg

        return res

    def __getitem__(self, item: str):
        return self.__getattr__(item)
