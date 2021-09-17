"""

"""
import textwrap
from pprint import pformat
from typing import List, Union, Any

import copy
import json
import os
import pprint as pp
import sys
import warnings
from collections import namedtuple
from collections.abc import Iterable
from datetime import timedelta
from itertools import chain
from typing import Any, overload, TypeVar, Optional, List, Union
from accelerate.kwargs_handlers import KwargsHandler
from accelerate.utils import RNGType
import fire
import torch

from lumo.base_classes.attr import attr
from lumo.base_classes.errors import BoundCheckError, NewParamWarning
from lumo.base_classes.params_vars import OptimBuilder, OptimMixin
from lumo.calculate import schedule
from lumo.utils import safe_io as io

arange_param = namedtuple('arange_param', ['default', 'left', 'right'], defaults=[None, float('-inf'), float('inf')])
choice_param = namedtuple('choice_param', ['default', 'choices'], defaults=[None, []])
default_param = namedtuple('default_param', ['default', 'warn'], defaults=[True])


class BaseParams:
    """
    Params make it easy to get/set/load/dump your config, if you use easy_dict before, you can see Params as a easy_dict ppplus.

    Notes:
        Variable name starts with `_` is not recommeded, because of the special inner implement, variable name starts with `_`
        will be ignored when serialize Params instnace, you will loss the value of these variables.

        If you really want to define some special variable to make it look different from others, you can capitalize it or ends with `_`.
    """

    def __init__(self):
        self._param_dict = attr()
        self._namespace = attr()
        self._repeat = -1
        self._constrain = {}
        self._lock = False

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        1. check constrain
        2. check if is default and not exists
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if isinstance(value, (arange_param, choice_param)):
                self._constrain[name] = value
                value = value.default
            else:
                self._check(name, value)

            if isinstance(value, default_param):  # 设置默认值
                # only set default param when name not exists
                if name not in self._param_dict:
                    if value.warn:
                        warnings.warn(
                            "'{}' is a new param,please check your spelling. It's more recommended to define in advance.".format(
                                name))
                    value = value.default
                    self._param_dict[name] = value
            else:
                self._param_dict[name] = value

            res = self._namespace.setdefault(self.__class__, [])
            res.append(name)

    def __getattr__(self, item):
        if item not in self._param_dict and self._lock:
            raise AttributeError(item)
        return self._param_dict.__getattr__(item)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            self._param_dict.pop(name)

    def __delitem__(self, key):
        key = str(key)
        self.__delattr__(key)

    def __contains__(self, item):
        return item in self._param_dict

    def __getstate__(self):
        return {
            '_param_dict': self._param_dict,
            '_repeat': self._repeat,
            '_lock': self._lock,
            '_bound': self._constrain,
            '_namespace': self._namespace,
        }

    def __setstate__(self, d):
        self._param_dict = d['_param_dict']
        self._repeat = d['_repeat']
        self._lock = d['_lock']
        self._constrain = d['_bound']
        self._namespace = d['_namespace']

    def __repr__(self):
        dynamic_propertys = [(k, io.safe_getattr(self, k, None)) for k in self.__dir__() if
                             isinstance(getattr(self.__class__, k, None), property)]

        dynamic_propertys = [(k, v, f'{type(v).__name__}') for k, v in dynamic_propertys]

        def _arg_to_str(k, v):
            res = self._constrain.get(k, None)
            if res is not None:
                return f'{res}, {type(v).__name__}'
            return f'{type(v).__name__}'

        args = [(k, v) for k, v in
                chain(self._param_dict.items())]
        args = [(k, v, _arg_to_str(k, v)) for k, v in args]

        args_str = BaseParams.safe_param_repr(args)

        if len(dynamic_propertys) > 0:
            property_str = BaseParams.safe_param_repr(dynamic_propertys)
            return "{}.Space".format(
                self.__class__.__name__) + '(\n' + args_str + '\n    # @property\n' + property_str + '\n)'
        return "{}.Space".format(self.__class__.__name__) + '(\n' + args_str + '\n)'

    __str__ = __repr__

    def __eq__(self, other):
        """
        equal function only compare params, and will ignore other Params hidden variables, including `_repeat`, `_lock`, and
        `_bound`, so if param_1 equals to param_2, it means all key-value (need to be hashable) pair in these two Params
        instance is equal.
        """
        if isinstance(other, BaseParams):
            return self.hash() == other.hash()
        return False

    def _check(self, name, value):
        if isinstance(value, default_param):
            value = value.default
        if name not in self._constrain:
            return True
        bound = self._constrain[name]
        if isinstance(bound, arange_param) and not (bound.left <= value and value <= bound.right):
            raise BoundCheckError(f"value of param '{name}' should in range [{bound.left}, {bound.right}].")
        elif isinstance(bound, choice_param) and value not in bound.choices:
            raise BoundCheckError(f"value of param '{name}' should in values {bound.choices}.")

    @staticmethod
    def _safe_repr(values: Any) -> str:
        return pformat(values)

    @staticmethod
    def _padding_mod(st: str, offset=7, mod=4):
        """
        123 \\
        1   \\
        12312341    \\
        1231
        Args:
            strs:
            mod:

        Returns:

        """
        size = len(st)
        if size < offset:
            return st.ljust(offset, ' ')

        mnum = mod - len(st) % mod
        # if mnum == 0:
        #     mnum = mod
        return st.ljust(size + mnum, ' ')

    @staticmethod
    def safe_param_repr(values: List[tuple], level=1) -> str:
        """

        Args:
            values:
            level:

        Returns:

        """
        res = [(f"{k}={BaseParams._safe_repr(v)},", anno) for k, v, anno in values]

        # res = textwrap.fill('\n'.join(res))
        res = '\n'.join([BaseParams._padding_mod(i, offset=16, mod=4) + f'  # {anno}' for i, anno in res])

        return textwrap.indent(res, '    ')

    @classmethod
    def Space(cls, **kwargs):
        return cls().from_dict(kwargs)

    def copy(self):
        res = self.__class__()
        res._param_dict = self._param_dict.copy()
        res._repeat = self._repeat
        res._bound = copy.deepcopy(self._constrain)
        res._lock = self._lock
        return res

    def arange(self, default, left=float("-inf"), right=float("inf")) -> arange_param:
        """
        Make sure some value is into some range.

        Examples:
            params.batch_size = params.arange(20,10,100)
            print(params.batch_size) # will print '20' as default.
            params.batch_size = 300 # will raise an Exception
            params.batch_size = 50
            print(params.batch_size) # will print 50

        Args:
            k: key of the value
            default: default value
            left: left interval
            right: right interval

        Returns:
            arange_param(default, left, right)
        """
        if left < default and default < right:
            return arange_param(default, left, right)
        else:
            raise BoundCheckError(f"value {default}' should in range [{left}, {right}].")

    def choice(self, *choices) -> choice_param:
        """
        Make sure some value is into some limited values.

        Examples:
            params.dataset = params.choice('cifar10','cifar100')
            print(params.dataset) # will print 'cifar10' as default.
            params.dataset = 'mnist' # will raise an Exception
            params.dataset = 'cifar100'
            print(params.dataset) # will print 'cifar100'

        Args:
            k: key of the value
            *choices: value can be used for key

        Returns:
            choice_param(choices[0], choices)


        """
        return choice_param(choices[0], choices)

    def default(self, value: Any = None, warn=True) -> default_param:
        """
        Set a default value to a key. This default value will be set only when `key` doesn't exists in Params.

        Examples:
        ```
        params.margin = 3
        params.margin = params.default(5,True)
        params.non_exists = params.default(0.3,True)
        print(params.margin)
        print(params.non_exists)
        ```

        Args:
            value: default value
            warn: warn if default value when set this default value, to warn user set this value manully in advance.
                default is True.

        Returns:
            default_param(value, warn)
        """
        return default_param(value, warn)

    def from_args(self):
        """
        Load key-value from command line arguments (based on facebook-Fire).


        Examples:
            python demo.py --k=12 --v=qwe

            # demo.py
            params.from_args()
            print(params.k) # will give `12` int object
            print(params.v) # will give `qwe` string object
        """

        def func(**kwargs):
            if '_help' in kwargs:
                print(self)
                exit()
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
                            k), NewParamWarning)
                self[k] = v

        fire.Fire(func)
        self.iparams()
        return self

    def from_yaml(self, fn):
        """
        Read params from yaml file, if file path `fn` not exist or some Exceptions are raised during load, the program won't be terminal
        but error messages will be printed in stderr.

        Args:
            fn: file path of the yaml file
        """
        if os.path.exists(fn):
            try:
                import yaml
                with open(fn, encoding='utf-8') as r:
                    res = yaml.safe_load(r)
                    for k, v in res.items():
                        self[k] = v
            except ImportError as e:
                print(
                    "from_yaml() operation will be ignored, cause you havn't install yaml, use `pip install yaml` to install it")
            except Exception as e:
                print('from_yaml() operation will be ignored, cause:')
                print(e, file=sys.stderr)
        else:
            print(f'{fn} not exists, ignored load params from yaml file, please verify the path.', file=sys.stderr)
        self.iparams()
        return self

    def from_json(self, fn):
        """
        Read params from json file, if file path `fn` not exist or some Exceptions are raised during load, the program won't be terminal
        but error messages will be printed in stderr.

        Args:
            fn: file path of the yaml file
        """
        if os.path.exists(fn):
            try:
                with open(fn, encoding='utf-8') as r:
                    res = json.load(r)
                    for k, v in res.items():
                        self[k] = v
            except Exception as e:
                print('from_json() operation will be ignored, cause:')
                print(e, file=sys.stderr)
        else:
            print(f'{fn} not exists, ignored load params from json file, please verify the path', file=sys.stderr)

        self.iparams()
        return self

    def from_dict(self, dic: dict):
        """Alias of update()"""
        for k, v in dic.items():
            self[k] = v

        self.iparams()
        return self

    def to_json(self, fn: str):
        """
        Save the params object to a disk file.

        Args:
            fn: a string object contraining a file name.

        Notes:
            - It's better to use '.json' extension.
            - Some key-value pair that cannot be serialized may be ignored
        """
        io.dump_json(self.inner_dict().jsonify(), fn)

    def items(self):
        """Like dict.items()"""
        return self._param_dict.items()

    def keys(self):
        """Like dict.keys()"""
        return self._param_dict.keys()

    def update(self, dic: dict):
        """Like dict.update()"""
        self._param_dict.update(dic)
        return self

    def hash(self) -> str:
        """
        Return a hash string of all key and values. See attr.hash() for details.
        """
        return self._param_dict.hash()

    def inner_dict(self) -> attr:
        """Return the inner attr, which is a dict-like object that saves all params."""
        return self._param_dict

    def get(self, k, default=None):
        """Like dict.get()"""
        if k in self:
            return self[k]
        else:
            return default

    def replace(self, **kwargs):
        """Alias of update()"""
        return self.update(kwargs)

    def contains(self, key: str):
        """Like dict.contains()"""
        return key in self

    def merge(self, *params: 'BaseParams'):
        """Merge other params values"""
        for param in params:
            self._param_dict.update(param._param_dict)

    def iparams(self):
        pass


class DistributionParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.backend = 'nccl'
        self.world_size = 1
        self.num_nodes = 1
        self.local_rank = -1

    @property
    def init_method(self):
        from lumo.proc.network import find_free_network_port
        port = find_free_network_port()
        return f'tcp://localhost:{port}'

    @overload
    def init_process_group(self, backend,
                           init_method=None,
                           timeout=timedelta(minutes=30),
                           world_size=1,
                           rank=-1,
                           store=None,
                           group_name=''):
        pass

    def init_process_group(self, **kwargs):
        self.init_process_group_args = (kwargs)
        return self.init_process_group_args


class OptimParams(BaseParams, OptimMixin):

    def __init__(self):
        super().__init__()


class Params(BaseParams):
    OPTIM = OptimParams()

    # DataLoaderParams = DataLoaderParams
    class SCHE:
        Cos = schedule.CosScheduler
        Linear = schedule.LinearScheduler
        Log = schedule.LogScheduler
        Exp = schedule.ExpScheduler
        Power = schedule.PowerDecayScheduler
        Const = schedule.ConstantScheduler

        PeriodCos = schedule.PeriodCosScheduler
        PeriodHalfCos = schedule.PeriodHalfCosScheduler
        PeriodLinear = schedule.PeriodLinear
        PeriodTriangle = schedule.PeriodTriangleScheduler

        List = schedule.SchedulerList

    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.eidx = 0
        self.idx = 0
        self.global_step = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.stage = self.choice('init', 'train', 'test', 'val')


ParamsType = TypeVar('ParamsType', bound=Params)


class AccelerateParams(BaseParams):

    def __init__(self):
        super().__init__()
        self.device_placement: bool = True
        self.split_batches: bool = False
        self.fp16: bool = None
        self.cpu: bool = False
        self.rng_types: Optional[List[Union[str, RNGType]]] = None
        self.kwargs_handlers: Optional[List[KwargsHandler]] = None


def disable_commit():
    params = BaseParams()
    params.nocommit = True
    return params


def use_prerain(pretrain=True, pretrain_path=None):
    params = BaseParams()
    params.pretrain = pretrain
    params.pretrain_path = pretrain_path
    return params

# class optims:
#     @staticmethod
#     def sgd():
#         pass
