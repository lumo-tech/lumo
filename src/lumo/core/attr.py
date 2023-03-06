"""
easydict, support dic['a.b'] to refer dic.a.b
"""
from collections import OrderedDict
from typing import List


class Attr(OrderedDict):
    """
    A subclass of OrderedDict that allows you to access its elements via dot notation.

    This class overrides the __setattr__, __setitem__, __getattr__, and __getitem__ methods to provide the
    dot notation functionality.
    """

    def __setattr__(self, key: str, value):
        set_item_iterative(self, key.split('.'), value)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Key in attr must be str')
        set_item_iterative(self, key.split('.'), value)

    def __getattr__(self, key: str):
        try:
            res = get_item_iterative(self, key.split('.'))
        except KeyError:
            res = Attr()
            set_item_iterative(self, key.split('.'), res)

        return res

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError('Key in attr must be str')
        return get_item_iterative(self, key.split('.'))


def safe_update_dict(src: dict, kwargs: dict, assert_type=False):
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


def walk_dict(dic: dict, root=None):
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


def set_item_iterative(dic: dict, keys: List[str], value):
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
        if isinstance(value, dict):
            for ks, v in walk_dict(value):
                set_item_iterative(dic, [*keys, *ks], v)
        else:
            OrderedDict.__setitem__(dic, keys[0], value)
    else:
        try:
            nex = OrderedDict.__getitem__(dic, keys[0])
            if not isinstance(nex, dict):
                raise ValueError(keys[0], nex)
            # dict.__setitem__(dic, keys[0], nex)
        except KeyError:
            nex = Attr()
            OrderedDict.__setitem__(dic, keys[0], nex)

        set_item_iterative(nex, keys[1:], value)


def get_item_iterative(dic: dict, keys: List[str]):
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
        return OrderedDict.__getitem__(dic, keys[0])
    else:
        nex = OrderedDict.__getitem__(dic, keys[0])
        if isinstance(nex, dict):
            return get_item_iterative(nex, keys[1:])
        else:
            raise KeyError(keys)
