"""

"""
import copy
import json
import os
import pprint as pp
import warnings
from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta
from typing import Any, overload

import fire
import torch

from ..base_classes.attr import attr
from ..base_classes.defaults import default
from ..base_classes.errors import BoundCheckError, NewParamWarning
from ..base_classes.params_vars import ParamsFactory, OptimParams, OptimMixin
from ..utils.environ import ENVIRON_
from ..calculate import schedule
from ..decorators.deprecated import deprecated


class BaseParams(OptimMixin):
    ENV = ENVIRON_

    class SCHE:
        Cos = schedule.CosSchedule
        Linear = schedule.LinearSchedule
        Log = schedule.LogSchedule
        Exp = schedule.ExpSchedule
        Power = schedule.PowerDecaySchedule
        Const = schedule.ConstantSchedule

        PeriodCos = schedule.PeriodCosSchedule
        PeriodHalfCos = schedule.PeriodHalfCosSchedule
        PeriodLinear = schedule.PeriodLinear
        PeriodTriangle = schedule.PeriodTriangleSchedule

        List = schedule.ScheduleList

    def __init__(self):
        self._param_dict = attr()
        self._repeat = None
        self._bound = {}
        self._lock = False
        # self._bind = defaultdict(list)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __contains__(self, item):
        return item in self._param_dict

    def __getstate__(self):
        return {
            '_param_dict': self._param_dict,
            '_repeat': self._repeat,
            # '_bound': self._bound,
            '_lock': self._lock,
            # '_bind': self._bind,
        }

    def __setstate__(self, d):
        self._param_dict = d['_param_dict']
        self._repeat = d['_repeat']
        # self._bound = d['_bound']
        self._lock = d['_lock']
        # self._bind = d['_bind']

    def __setattr__(self, name: str, value: Any) -> None:
        """
        1. check constrain
        2. check if is default and not exists
        3. check bind
        """
        from ..base_classes.defaults import default
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            res = self._check(name, value)
            if res is not None and not res:
                raise BoundCheckError("param '{}' checked failed.".format(name))

            if isinstance(value, default):  # 设置默认值
                # only set default value when name not exists
                if name not in self._param_dict:
                    if value.warn:
                        warnings.warn(
                            "'{}' is a new param,please check your spelling. It's more recommended to define in advance.".format(
                                name))
                    value = value.default
                    self._param_dict[name] = value
            else:
                self._param_dict[name] = value
            # if name in self._bind:
            # for _v, _bind_k, _bind_v in self._bind[name]:
            #     if callable(_v):
            #         _bind_v = _v(value)
            #         self.__setattr__(_bind_k, _bind_v)
            #     elif _v == value:
            #         self.__setattr__(_bind_k, _bind_v)

    def __getattr__(self, item):
        return self._param_dict.__getattr__(item)

    def __repr__(self):
        return "{}".format(self.__class__.__name__) + pp.pformat([(k, v) for k, v in self._param_dict.items()])

    __str__ = __repr__

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            self._param_dict.pop(name)

    def __delitem__(self, key):
        key = str(key)
        self.__delattr__(key)

    def __eq__(self, other):
        if isinstance(other, BaseParams):
            return self.hash() == other.hash()
        return False

    def _check(self, name, value):
        if isinstance(value, default):
            value = value.default
        if name in self._bound:
            self._bound[name](value)

    def initial(self):
        pass

    def arange(self, k, default, left=float("-inf"), right=float("inf")):
        """
        constrain value within a continuous interval.
        Args:
            k: key of the value
            default: default value
            left: left interval
            right: right interval

        Returns:
            default value

        Notes:
        ---------
        you can directly call this function to create the key with the constrained value, but it's better to
        receive this value so that IDE can recognizes it and provides auto-complete.
        """

        def check(x):
            if x < left or x > right:
                raise BoundCheckError("param '{}' must be within range({}, {})".format(k, left, right))

        self._bound[k] = check
        self[k] = default
        return default

    def choice(self, k, *choices):
        """
        constrain value within a discrete set.
        Args:
            k: key of the value
            *choices: value can be used for key

        Returns:
            the first value of choices

        """

        def check(x):
            if x not in choices:
                raise BoundCheckError("param '{}' is enum of {}".format(k, choices))

        self._bound[k] = check
        self[k] = choices[0]
        return choices[0]

    def lambda_bound(self, k, default, check_lmd):
        """"""
        pass
        # self._bound[k] = check_lmd
        # self[k] = default
        # return default

    @deprecated('1.5.1', '1.6', 'Call initial() to create dynamic parameter is a better choice.')
    def bind(self, k, v, bind_k, bind_v):
        """
        Link the change in the value of one key with another.
        When params[k] is set to v, params[bind_k] will be set to bind_v automatic
        """
        # self._bind[k].append((v, bind_k, bind_v))
        pass

    @deprecated('1.5.1', '1.6', 'Call initial() to create dynamic parameter is a better choice.')
    def dynamic_bind(self, k, bind_k, dynamic_func):
        # self._bind[k].append((dynamic_func, bind_k, None))
        pass

    def grid_search(self, key, iterable: Iterable):
        """"""
        for v in iterable:
            res = self._copy()
            res[key] = v
            yield res

    def _copy(self):
        res = BaseParams()
        res._param_dict = copy.copy(self._param_dict)
        res._repeat = copy.copy(self._repeat)
        # res._bound = copy.copy(self._bound)
        res._lock = copy.copy(self._lock)
        # res._bind = copy.copy(self._bind)
        return res

    def grid_range(self, count):
        for i in range(count):
            res = self._copy()
            res._repeat = i
            yield res

    def from_args(self):
        """
        从命令行参数中设置参数值

        可选参数：
            _json, 接收一个json来作为该params的参数
            _help, 输出该Params已设置的参数

        Returns:

        """

        def func(**kwargs):
            if '_help' in kwargs:
                print(self)
                return

            if '_json' in kwargs:
                self.from_json(kwargs['_json'])
                return

            for k, v in kwargs.items():
                try:
                    self[k]
                except:
                    warnings.simplefilter('always', NewParamWarning)
                    warnings.warn(
                        "'{}' is a new param,please check your spelling.\n it's more recommended to define in advance.".format(
                            k))
                self[k] = v

        fire.Fire(func)
        return self

    def from_json(self, fn):
        """
        从 json 中获取参数值
        Args:
            fn:

        Returns:

        """
        if os.path.exists(fn):
            with open(fn, encoding='utf-8') as r:
                res = json.load(r)
                for k, v in res.items():
                    self[k] = v
        return self

    def to_json(self, fn: str):
        """
        以json格式保存文件，注意，该方法保存的json内容基于 attr 的jsonify() 方法，不可序列化的格式无法被保存
        Args:
            fn:

        Returns:

        """
        with open(fn, 'w', encoding='utf-8') as w:
            json.dump(self.inner_dict.jsonify(), w, indent=2)

    def items(self):
        return self._param_dict.items()

    def keys(self):
        for k in self._param_dict:
            yield k

    def update(self, dic: dict):
        """

        Args:
            dic:

        Returns:

        """
        for k, v in dic.items():
            self._param_dict[k] = v

        return self

    def hash(self) -> str:
        """
        返回对参数的定义顺序及其相应值的一个hash，理论上，两个Param 对象的hash方法返回的参数相同，
        则两者具有相同的参数和参数值

        Returns:

        """
        return self._param_dict.hash()

    def lock(self):
        """
        锁定当前配置，如果当前配置未 lock，那么当尝试获取未分配的参数时候，会返回一个空的字典
        如果锁定，则在尝试获取未分配参数时，会抛出 AttributeError(key)
        Returns:

        """
        self._lock = True

    @property
    def inner_dict(self) -> attr:
        return self._param_dict

    def get(self, k, default=None):
        """
        获取某值，如果不存在，则返回默认值
        Args:
            k:
            default:

        Returns:

        """
        if k in self:
            return self[k]
        else:
            return default

    @staticmethod
    def default(value: Any = None, warn=False):
        """
        默认值，分配值时，仅当当前key没有分配时，分配该值作为键值。否则，该值会被忽略

        Examples:
        >>> params.margin = params.default(0.5,True)
        >>> params.margin = params.default(0.3,True)
        >>> print(params.margin)

        Args:
            value: 要设置的值
            warn: 当设置默认值时，抛出警告

        Returns:
            default(value, warn)
        """
        from ..base_classes.defaults import default
        return default(value, warn)

    def create_schedule(self, schedule_type, start, end, **kwargs):
        return ParamsFactory

    def replace(self, **kwargs):
        self.update(kwargs)
        return self

    def contains(self, key: str):
        return key in self


class DistributionParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.backend = 'nccl'
        self.distributed = False
        self.world_size = -1
        self.local_rank = -1  # if not -1, means will use
        self.init_method = 'env://'


class Params(BaseParams):
    Attr = attr

    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.eidx = 1
        self.idx = 0
        self.global_step = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_ids = []
        self.dataset = None
        self.architecture = None
        self.optim = None  # type:OptimParams
        self.git_commit = True
        self.tmp_dir = None  # type:str # set TMPDIR environment

        self.distributed = False
        self.world_size = -1
        self.local_rank = -1  # if not -1, means will use
        self.init_method = 'env://'

    def enable_distribution(self):
        pass

    @overload
    def init_process_group(self, backend,
                           init_method=None,
                           timeout=timedelta(minutes=30),
                           world_size=-1,
                           rank=-1,
                           store=None,
                           group_name=''):
        pass

    def init_process_group(self, *args, **kwargs):
        self.init_process_group_args = (args, kwargs)
        return self.init_process_group_args
