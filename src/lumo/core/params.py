import json
import os.path
import sys
import textwrap
from pprint import pformat
from typing import Any, List, NewType

import fire
from joblib import hash
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, DictKeyType
from omegaconf._utils import _ensure_container

from .attr import safe_update_dict, set_item_iterative
from .raises import BoundCheckError

# arange_param = namedtuple('arange_param', ['default', 'left', 'right'], defaults=[None, float('-inf'), float('inf')])
# choice_param = namedtuple('choice_param', ['default', 'choices'], defaults=[None, []])

__all__ = ['BaseParams', 'Params', 'ParamsType']


class Arange:
    """A class representing a range of numeric values with a default, left and right boundaries.

    Attributes:
        default: The default value of the range. Defaults to None.
        left: The left boundary of the range. Defaults to positive infinity.
        right: The right boundary of the range. Defaults to positive infinity.
    """

    def __init__(self, default=None, left=float('inf'), right=float('inf')):
        self.default = default
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Arange: {self.default}, [{self.left}, {self.right}] "


class Choices:
    """A class representing a list of choices with a default value.

    Attributes:
        default: The default value of the list. Defaults to None.
        choices: A list of values representing the available choices. Defaults to an empty list.
    """

    def __init__(self, default=None, choices=None):
        if choices is None:
            choices = []
        self.default = default
        self.choices = choices

    def __repr__(self):
        return f"Choice: [{self.default}], {self.choices}"


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
    Returns a string representation of a list of tuples containing parameter names and their values.
    The resulting string can be safely included in a function signature or call, as it correctly formats the parameters.

    Args:
        values: A list of tuples containing parameter names and their values.
        level: An integer representing the level of indentation.

    Returns:
        A string representation of the input parameters, formatted with correct indentation and line breaks.
    """
    res = [(f"{k}={_safe_repr(v)},", anno) for k, v, anno in values]

    # res = textwrap.fill('\n'.join(res))
    res = '\n'.join([_padding_mod(i, offset=16, mod=4) + f'  # {anno}' for i, anno in res])

    return textwrap.indent(res, '    ')


class BaseParams(DictConfig):
    def __init__(self):
        super().__init__({}, flags={'no_deepcopy_set_nodes': True})
        self.__dict__["_prop"] = {}

    def __setattr__(self, key: str, value: Any) -> None:
        if key != '_prop':
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
            if isinstance(value, (Arange, Choices)):
                self._prop.setdefault('constrain', {})[key] = value
                value = value.default

            if key in self._prop.setdefault('constrain', {}):
                self._check(key, value)

        super().__setitem__(key, value)

    def __getattr__(self, key: str) -> Any:
        res = super().__getattr__(key)
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

    def copy(self):
        copied = self.__class__()
        copied.from_dict(super(BaseParams, self).copy())
        return copied

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

    def safe_update(self, dic, assert_type=True):
        """
        Merge `dict` object into the config object, safely updating the values.

        Args:
            dic: `dict` object to update
            assert_type: If True, enforce that the type of values in `dic` matches the current config.

        Returns:
            None
        """
        self.update(
            safe_update_dict(self.to_dict(), dic, assert_type=assert_type)
        )

    def from_dict(self, dic: dict):
        """
        Update the config object from a dictionary.

        Args:
            dic: `dict` object to update

        Returns:
            updated `self` object
        """
        self.safe_update(dic)
        return self

    def from_kwargs(self, **kwargs):
        """
        Update the config object from keyword arguments.

        Args:
            **kwargs: key-value pairs to update in the config object

        Returns:
            updated `self` object
        """
        return self.from_dict(kwargs)

    def from_json(self, file):
        """
        Update the config object from a JSON file.

        Args:
            file: path to the JSON file

        Returns:
            updated `self` object
        """
        self.safe_update(json.loads(Path(file).read_text()), assert_type=True)
        return self

    def from_yaml(self, file):
        """
        Update the config object from a YAML file.

        Args:
            file: path to the YAML file

        Returns:
            updated `self` object
        """
        self.safe_update(dict(OmegaConf.load(file)), assert_type=True)
        return self

    def from_args(self, argv: list = None):
        """
        Update the config object from command line arguments.

        Args:
            argv: list of command line arguments (default: None)

        Returns:
            updated `self` object
        """
        if argv is None:
            argv = sys.argv

        def func(**kwargs):
            if 'help' in kwargs:
                print(self)
                exit()
                return

            config = kwargs.get('config')
            if config is None:
                config = kwargs.get('c')
            if config is not None and isinstance(config, str) and os.path.exists(config):
                self.from_yaml(config)

            dic = {}
            for k, v in kwargs.items():
                set_item_iterative(dic, k.split('.'), v)
            self.safe_update(dic)

        fire.Fire(func, command=argv)
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
        """
        Convert this configuration to a dictionary.

        Returns:
            dict: The configuration as a dictionary.
        """
        cfg = _ensure_container(self)
        container = OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
        return container

    def to_json(self, file=None):
        """
        Convert this configuration to a JSON string.

        Args:
            file (str or Path, optional): If specified, the JSON string will be written to a file at the given path.

        Returns:
            str or None: The JSON string, or None if file is specified.
        """
        info = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if file is None:
            return info
        return Path(file).write_text(info, encoding='utf-8')

    def to_yaml(self, file=None):
        """
        Convert this configuration to a YAML string.

        Args:
            file (str or Path, optional): If specified, the YAML string will be written to a file at the given path.

        Returns:
            str or None: The YAML string, or None if file is specified.
        """
        info = OmegaConf.to_yaml(self)
        if file is None:
            return info
        return Path(file).write_text(info, encoding='utf-8')

    @classmethod
    def Space(cls, **kwargs):
        """
        Create a configuration object from keyword arguments.

        Args:
            **kwargs: The configuration values.

        Returns:
            BaseParams: The new configuration object.
        """
        return cls().from_dict(kwargs)

    def __hash__(self):
        """
        Calculate the hash value of this configuration.

        Returns:
            int: The hash value.
        """
        return int(self.hash(), 16)

    def hash(self) -> str:
        """
        Calculate the hash value of this configuration.

        Returns:
            str: The hash value.
        """
        return hash(self.to_dict())

    def iparams(self):
        pass

    @classmethod
    def init_from_kwargs(cls, **kwargs):
        """
        Create a configuration object from keyword arguments.

        Args:
            **kwargs: The configuration values.

        Returns:
            BaseParams: The new configuration object.
        """
        return cls().from_dict(kwargs)


class Params(BaseParams):
    pass


ParamsType = NewType('ParamsType', Params)
