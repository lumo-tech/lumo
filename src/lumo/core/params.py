import json
import sys
import textwrap
from pprint import pformat
from typing import Any, List, NewType

import fire
from joblib import hash
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, DictKeyType
from omegaconf._utils import _ensure_container

from .raises import BoundCheckError, NewParamWarning

# arange_param = namedtuple('arange_param', ['default', 'left', 'right'], defaults=[None, float('-inf'), float('inf')])
# choice_param = namedtuple('choice_param', ['default', 'choices'], defaults=[None, []])

__all__ = ['BaseParams', 'Params', 'ParamsType']


class Arange:
    def __init__(self, default=None, left=float('inf'), right=float('inf')):
        self.default = default
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Arange: {self.default}, [{self.left}, {self.right}] "


class Choices:
    def __init__(self, default=None, choices=None):
        if choices is None:
            choices = []
        self.default = default
        self.choices = choices

    def __repr__(self):
        return f"Choice: [{self.default}], {self.choices}"


def _get_item(dic, keys: List[str]):
    if len(keys) == 1:
        return DictConfig.__getitem__(dic, keys[0])
    else:
        nex = DictConfig.__getitem__(dic, keys[0])
        if isinstance(nex, (dict, DictConfig)):
            return _get_item(nex, keys[1:])
        else:
            raise KeyError(keys)


def _set_item(dic, keys: List[str], value):
    if len(keys) == 1:
        if isinstance(value, dict):
            value = dic(value)
        DictConfig.__setitem__(dic, keys[0], value)
    else:
        try:
            nex = _get_item(dic, keys[:1])
        except KeyError:
            nex = DictConfig({})
            DictConfig.__setitem__(dic, keys[0], nex)
        _set_item(nex, keys[1:], value)


def _safe_repr(values: Any) -> str:
    return pformat(values)


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


def safe_param_repr(values: List[tuple], level=1) -> str:
    """

    Args:
        values:
        level:

    Returns:

    """
    res = [(f"{k}={_safe_repr(v)},", anno) for k, v, anno in values]

    # res = textwrap.fill('\n'.join(res))
    res = '\n'.join([_padding_mod(i, offset=16, mod=4) + f'  # {anno}' for i, anno in res])

    return textwrap.indent(res, '    ')


class BaseParams(DictConfig):
    def __init__(self):
        super().__init__({}, flags={'no_deepcopy_set_nodes': True})
        # self._set_flag('no_deepcopy_set_nodes', True)
        self.__dict__["_prop"] = {}

    def __setattr__(self, key: str, value: Any) -> None:
        if key != '_prop':
            # if isinstance(value, BaseParams):
            #     self._prop.setdefault('key_type', {})[key] = type(value)

            if isinstance(value, (Arange, Choices)):
                res = self._prop.get('constrain', {})
                res[key] = value
                self._prop['constrain'] = res
                value = value.default

            if key in self._prop.setdefault('constrain', {}):
                self._check(key, value)

        super().__setattr__(key, value)

    def __setitem__(self, key: DictKeyType, value: Any) -> None:
        if key != '_prop':
            # if isinstance(value, BaseParams):
            #     self._prop.setdefault('key_type', {})[key] = type(value)

            if isinstance(value, (Arange, Choices)):
                self._prop.setdefault('constrain', {})[key] = value
                value = value.default

            if key in self._prop.setdefault('constrain', {}):
                self._check(key, value)

        super().__setitem__(key, value)

    def __getattr__(self, key: str) -> Any:
        res = super().__getattr__(key)
        # key_type = self._prop.setdefault('key_type', {}).get(key, None)
        # if key_type is not None:
        #     res = key_type.from_kwargs(**res)
        return res

    def _check(self, name, value):
        bound = self._prop['constrain'][name]
        if isinstance(bound, Arange) and not (bound.left <= value and value <= bound.right):
            raise BoundCheckError(
                f"value of param '{name}' should in range [{bound.left}, {bound.right}], but got {value}")
        elif isinstance(bound, Choices) and value not in bound.choices:
            raise BoundCheckError(f"value of param '{name}' should in values {bound.choices}, but got {value}")

    def __getitem__(self, key: DictKeyType) -> Any:
        return super().__getitem__(key)

    def __repr__(self):
        def _arg_to_str(k, v):
            res = self._prop.get('constrain', {}).get(k, None)
            if res is not None:
                return f'{res}, {type(v).__name__}'

            key_type = self._prop.get('key_type', {}).get(k, None)
            if key_type is not None:
                return f'{key_type.__name__}'

            return f'{type(v).__name__}'

        args = [(k, v) for k, v in self.items()]
        args = [(k, v, _arg_to_str(k, v)) for k, v in args]

        args_str = safe_param_repr(args)

        return "{}.Space".format(self.__class__.__name__) + '(\n' + args_str + '\n)'

    def arange(self, default, left=float("-inf"), right=float("inf")) -> Arange:
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
            return Arange(default, left, right)
        else:
            raise BoundCheckError(f"value {default}' should in range [{left}, {right}].")

    def choice(self, *choices) -> Choices:
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
        return Choices(choices[0], choices)

    def from_dict(self, dic: dict):
        for k, v in dic.items():
            self[k] = v
        return self

    def from_kwargs(self, **kwargs):
        return self.from_dict(kwargs)

    def from_json(self, file):
        self.update(json.loads(Path(file).read_text()))
        return self

    def from_yaml(self, file):
        self.update(OmegaConf.load(file))
        return self

    def from_args(self, argv: list = None):
        if argv is None:
            argv = sys.argv

        def func(**kwargs):
            if '_help' in kwargs:
                print(self)
                exit()
                return

            if '_json' in kwargs:
                self.from_json(kwargs['_json'])
                return

            for k, v in kwargs.items():
                # try:
                #     _get_item(self, k.split('.'))
                # except:
                # self[k] = v
                _set_item(self, k.split('.'), v)

        fire.Fire(func)
        return self

    def from_hydra(self, config_path, config_name):
        import hydra
        hydra.compose()

        @hydra.main(config_path=config_path, config_name=config_name)
        def inner(cfg):
            return cfg

        self.update(inner())
        return self

    def to_dict(self):
        cfg = _ensure_container(self)
        container = OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
        return container

    def to_json(self, file=None):
        info = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if file is None:
            return info
        return Path(file).write_text(info, encoding='utf-8')

    def to_yaml(self, file=None):
        info = OmegaConf.to_yaml(self)
        if file is None:
            return info
        return Path(file).write_text(info, encoding='utf-8')

    @classmethod
    def Space(cls, **kwargs):
        return cls().from_dict(kwargs)

    def __hash__(self):
        return int(self.hash(), 16)

    def hash(self) -> str:
        return hash(str(self))

    def iparams(self):
        pass

    @classmethod
    def init_from_kwargs(cls, **kwargs):
        return cls().from_dict(kwargs)


class Params(BaseParams):
    pass


ParamsType = NewType('ParamsType', Params)
