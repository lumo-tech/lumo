import json
import os.path
import sys
import textwrap
from pathlib import Path
from pprint import pformat
from typing import Any, List, NewType, MutableMapping

import fire
from joblib import hash
from omegaconf import DictConfig, OmegaConf, DictKeyType, ListConfig
from omegaconf._utils import _ensure_container

# from .attr import safe_update_dict, set_item_iterative
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
    """Return a formatted string representation of the input values.

    Args:
        values: Any type of input values to be formatted.

    Returns:
        A string representation of the input values, formatted using `pprint`.

    Raises:
        None.
    """
    return pformat(values)


def _padding_mod(st: str, offset=7, mod=4):
    """Pads a string with spaces to a length that is a multiple of a given modulus.

    Args:
        st: The input string to pad.
        offset: An integer specifying the minimum length of the output string. If the length of the input string is
            less than this value, spaces will be added to the end of the string to make it the desired length.
        mod: An integer specifying the modulus. The length of the output string will be a multiple of this value.

    Returns:
        A string that is a multiple of the given modulus and has a length of at least `offset`. If the length of the
        input string is less than `offset`, the output string will be padded with spaces to achieve the minimum length.
        If the length of the input string is already a multiple of the given modulus, the output string will have the
        same length as the input string.
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
    """
    A dictionary-like configuration object that supports parameter constraint validation.
    """

    def __init__(self):
        """
        Initializes a new instance of the BaseParams class.
        """
        super().__init__({}, flags={'no_deepcopy_set_nodes': True})
        self.__dict__["_prop"] = {}

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Sets an attribute value for the specified key.

        Args:
            key (str): The key of the attribute.
            value (Any): The value of the attribute.

        Raises:
            BoundCheckError: If the specified value is not within the specified bounds or choices.

        """
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
        """
        Sets a dictionary item value for the specified key.

        Args:
            key (DictKeyType): The key of the item.
            value (Any): The value of the item.

        Raises:
            BoundCheckError: If the specified value is not within the specified bounds or choices.

        """
        if key != '_prop':
            if isinstance(value, (Arange, Choices)):
                self._prop.setdefault('constrain', {})[key] = value
                value = value.default

            if key in self._prop.setdefault('constrain', {}):
                self._check(key, value)

        super().__setitem__(key, value)

    def __getattr__(self, key: str) -> Any:
        """
        Gets an attribute value for the specified key.

        Args:
            key (str): The key of the attribute.

        Returns:
            Any: The value of the attribute.

        """
        res = super().__getattr__(key)
        return res

    def _check(self, name, value):
        """
        Checks if the specified parameter value is within the specified bounds or choices.

        Args:
            name (str): The name of the parameter.
            value (Any): The value of the parameter.

        Raises:
            BoundCheckError: If the specified value is not within the specified bounds or choices.

        """
        bound = self._prop['constrain'][name]
        if isinstance(bound, Arange) and not (bound.left <= value and value <= bound.right):
            raise BoundCheckError(
                f"value of param '{name}' should in range [{bound.left}, {bound.right}], but got {value}")
        elif isinstance(bound, Choices) and value not in bound.choices:
            raise BoundCheckError(f"value of param '{name}' should in values {bound.choices}, but got {value}")

    def __getitem__(self, key: DictKeyType) -> Any:
        """
        Gets a dictionary item value for the specified key.

        Args:
            key (DictKeyType): The key of the item.

        Returns:
            Any: The value of the item.

        """
        return super().__getitem__(key)

    def __repr__(self):
        """
        Returns a string representation of the BaseParams object.

        Returns:
            str: A string representation of the BaseParams object.

        """

        def _arg_to_str(k, v):
            """to str"""
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
        """
        Returns a copy of the BaseParams object.

        Returns:
            BaseParams: A copy of the BaseParams object.

        """
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

    def safe_update(self, dic, assert_type=False):
        """
        Merge `dict` object into the config object, safely updating the values.

        Args:
            dic: `dict` object to update
            assert_type: If True, enforce that the type of values in `dic` matches the current config.

        Returns:
            None
        """
        safe_update_dict(self, dic, assert_type=assert_type)

    def from_dict(self, dic: MutableMapping):
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
        self.safe_update(json.loads(Path(file).read_text()), assert_type=False)
        return self

    def from_yaml(self, file):
        """
        Update the config object from a YAML file.

        Args:
            file: path to the YAML file

        Returns:
            updated `self` object
        """
        self.safe_update(dict(OmegaConf.load(file)), assert_type=False)
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

        def func(*args, **kwargs):
            """function to process arg list"""
            if 'help' in kwargs:
                print(self)
                exit()
                return
            config = kwargs.get('config')
            if config is None:
                config = kwargs.get('c')

            if config is not None:
                if isinstance(config, str):
                    config = config.split(',')
                if isinstance(config, (list, ListConfig)):
                    for config_fn in config:
                        print('get', config_fn, 'done')
                        if not (isinstance(config_fn, str) and os.path.exists(config_fn)):
                            continue
                        if config_fn.endswith('yaml') or config_fn.endswith('yml'):
                            self.from_yaml(config_fn)
                        elif config_fn.endswith('json'):
                            self.from_json(config_fn)

            dic = BaseParams()
            for k, v in kwargs.items():
                set_item_iterative(dic, k.split('.'), v)
            self.safe_update(dic.to_dict())

        fire.Fire(func, command=argv)
        return self

    def from_hydra(self, config_path, config_name):
        """load from hydra config mode"""
        import hydra
        hydra.compose()

        @hydra.main(config_path=config_path, config_name=config_name)
        def inner(cfg):
            """inner function"""
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
        """Initialization method, mostly used in Trainer"""
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
    """A class representing parameters"""
    pass


ParamsType = NewType('ParamsType', Params)


def safe_update_dict(src: BaseParams, kwargs: dict, assert_type=False):
    """
   Updates the source dictionary with the key-value pairs from the kwargs dictionary in a safe manner.

   This function iterates over the items in the kwargs dictionary and updates the corresponding items in the
   source dictionary, making sure that the types of the values being updated match the types of the values
   already in the source dictionary.

   Args:
       src (dict): The dictionary to update.
       kwargs (dict): The dictionary containing the new key-value pairs to add to the source dictionary.
       assert_type (bool): A flag indicating whether to check that the types of the values being updated match
           the types of the values already in the source dictionary. Defaults to True.

   Returns:
       dict: The updated source dictionary.
   """
    for ks, v in walk_dict(kwargs):
        try:
            old_v = get_item_iterative(src, ks)
            if old_v is None or isinstance(old_v, type(v)) or not assert_type:
                set_item_iterative(src, ks, v)
            else:
                raise TypeError(ks, type(old_v), type(v))
        except KeyError:
            set_item_iterative(src, ks, v)
    return src


def walk_dict(dic: MutableMapping, root=None):
    """
    Recursively walks through a dictionary and yields keys and values in a flattened format.

    Args:
    - dic (dict): The dictionary to be walked through.
    - root (list): The root keys to be used in the resulting flattened format. Defaults to None.

    Yields:
    - A tuple containing a list of keys and a value. The list of keys is composed of the root keys and the current keys in the dictionary, split by '.' if there are any. The value is the corresponding value in the dictionary.

    Example:
        ```python
        d = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
        for k, v in walk_dict(d):
            print(k, v)
        # Output:
        # (['a', 'b'], 1)
        # (['a', 'c', 'd'], 2)
        # (['e'], 3)
        ```
    """
    if root is None:
        root = []
    for k, v in dic.items():
        if isinstance(v, dict):
            yield from walk_dict(v, [*root, *k.split('.')])
        else:
            yield [*root, *k.split('.')], v


def set_item_iterative(dic: BaseParams, keys: List[str], value):
    """
    Sets the value of a nested key in a dictionary using an iterative approach.

    Args:
        dic (dict): The dictionary to update.
        keys (List[str]): A list of keys representing the path to the nested key in the dictionary.
        value: The value to set for the nested key.

    Raises:
        ValueError: If a key in the path exists in the dictionary but the corresponding value is not a dictionary.

    """
    if len(keys) == 1:
        if isinstance(value, MutableMapping):
            for ks, v in walk_dict(value):
                set_item_iterative(dic, [*keys, *ks], v)
        else:
            dic.__setitem__(keys[0], value)
    else:
        try:
            nex = dic.__getitem__(keys[0])
            if not isinstance(nex, MutableMapping):
                raise ValueError(keys[0], nex)
            # dict.__setitem__(dic, keys[0], nex)
        except KeyError:
            nex = BaseParams()
            dic.__setitem__(keys[0], nex)

        set_item_iterative(nex, keys[1:], value)


def get_item_iterative(dic: MutableMapping, keys: List[str]):
    """
    Gets the value of a nested key in a dictionary using an iterative approach.

    Args:
        dic (dict): The dictionary to retrieve the value from.
        keys (List[str]): A list of keys representing the path to the nested key in the dictionary.

    Raises:
        KeyError: If the nested key does not exist in the dictionary.

    Returns:
        The value of the nested key in the dictionary.

    """
    if len(keys) == 1:
        return dic.__getitem__(keys[0])
    else:
        nex = dic.__getitem__(keys[0])
        if isinstance(nex, MutableMapping):
            return get_item_iterative(nex, keys[1:])
        else:
            raise KeyError(keys)
