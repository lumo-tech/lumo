"""
Safe means Exceptions won't be raised. When
 - call `dump_xxx` methods, True/False will be returned to indicate success/failed.
 - call `load_xxx` methods, None will be returned if something wrong happened.
"""
import json
import os
import pickle as _pickle
from contextlib import contextmanager
from io import FileIO

import torch
from joblib.numpy_pickle import dump as dump_nd, load as load_nd

dump_nd = dump_nd
load_nd = load_nd


def filter_unserializable_values(dic):
    """
    Recursively filter out unserializable values in a dictionary.

    Args:
        dic (dict): The dictionary to be filtered.

    Returns:
        The filtered dictionary.
    """
    for key, value in list(dic.items()):
        if isinstance(value, dict):
            filter_unserializable_values(value)
        elif isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], dict):
                    filter_unserializable_values(value[i])
                elif not json.dumps(value[i], default=lambda x: None):
                    value[i] = None
        elif json.dumps(value, default=lambda x: None) == 'null':
            dic[key] = None
    dic = {key: value for key, value in dic.items() if value is not None}
    return dic


def dump_json(obj, fn):
    """
    Dumps the given object to a JSON file at the given file path.

    Args:
        obj: The object to be dumped to JSON.
        fn (str): The file path to which the JSON data will be written.

    Notes:
        The JSON data will be written with an indentation of 2 spaces.
    """
    try:
        with open(fn, 'w', encoding='utf-8') as w:
            json.dump(obj, w, indent=2)
    except TypeError as e:
        raise TypeError(str(obj)) from e


def dump_yaml(obj, fn):
    """
    Dumps the given object to a YAML file at the given file path.

    Args:
        obj: The object to be dumped to YAML.
        fn (str): The file path to which the YAML data will be written.

    Notes:
        The YAML data will be written with default formatting options.
    """
    import yaml
    with open(fn, 'w', encoding='utf-8') as w:
        yaml.safe_dump(obj, w)


def dump_state_dict(obj, fn):
    """Saves a PyTorch state dictionary object to disk."""
    torch.save(obj, fn)


def load_json(fn):
    """
    Loads JSON data from the given file path and returns the resulting object.

    Args:
        fn: file name

    Returns:

    Raises:
        ValueError
    """
    try:
        with open(fn, 'r', encoding='utf-8') as r:
            return json.load(r)
    except json.JSONDecodeError as e:
        raise ValueError(f'Error in file {fn}') from e


def load_yaml(fn):
    """Loads YAML data from the given file path and returns the resulting object."""
    import yaml
    with open(fn, 'r', encoding='utf-8') as r:
        return yaml.safe_load(r)


def load_state_dict(fn: str, map_location='cpu'):
    """Loads a PyTorch model checkpoint from the specified file path and returns its state dictionary."""
    ckpt = torch.load(fn, map_location=map_location)
    return ckpt


def load_text(fn):
    """Loads text data from the given file path and returns it as a single string."""
    if not os.path.exists(fn):
        return ''
    with open(fn, 'r', encoding='utf-8') as r:
        return ''.join(r.readlines())


def dump_text(string: str, fn, append=False):
    """Write the given string to a file.

    Args:
        string (str): The string to write.
        fn (str): The filename to write to.
        append (bool, optional): If True, append the string to the file. Otherwise, overwrite the file. Defaults to False.

    Returns:
        str: The filename that was written to.
    """
    mode = 'w'
    if append:
        mode = 'a'
    with open(fn, mode, encoding='utf-8') as w:
        w.write(string)
    return fn


def safe_getattr(self, key, default=None):
    """Get an attribute of an object safely.

    Args:
        self (object): The object to get the attribute from.
        key (str): The name of the attribute to get.
        default (object, optional): The value to return if the attribute does not exist. Defaults to None.

    Returns:
        object: The value of the attribute if it exists, otherwise the default value.
    """
    try:
        return getattr(self, key, default)
    except:
        return default


def dump_pkl(obj, file, make_path=True, protocol=None, *, fix_imports=True):
    """Save an object to a pickle file.

    Args:
        obj (object): The object to save.
        file (str or FileIO): The filename or file object to save to.
        make_path (bool, optional): If True and file is a filename, create the directory path if it does not exist. Defaults to True.
        protocol (int, optional): The pickle protocol to use. Defaults to None.
        fix_imports (bool, optional): Whether to fix Python 2 to 3 pickle incompatibilities. Defaults to True.

    Raises:
        NotImplementedError: If the file type is not supported.

    """
    if isinstance(file, str):
        if make_path:
            os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
        file = open(file, 'wb')
        _pickle.dump(obj, file, protocol=protocol, fix_imports=fix_imports)
        file.close()
    elif isinstance(file, FileIO):
        _pickle.dump(obj, file, protocol=protocol, fix_imports=fix_imports)
    else:
        raise NotImplementedError("File type not supported.")


def load_pkl(file, *, fix_imports=True, encoding="ASCII", errors="strict"):
    """Load an object from a pickle file.

    Args:
        file (str or FileIO): The filename or file object to load from.
        fix_imports (bool, optional): Whether to fix Python 2 to 3 pickle incompatibilities. Defaults to True.
        encoding (str, optional): The character encoding to use. Defaults to "ASCII".
        errors (str, optional): The error handling scheme to use. Defaults to "strict".

    Returns:
        object: The object that was loaded from the file.

    Raises:
        NotImplementedError: If the file type is not supported.

    """
    if isinstance(file, str):
        file = open(file, 'rb')
        res = _pickle.load(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
        file.close()
        return res
    elif isinstance(file, FileIO):
        return _pickle.load(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
    else:
        raise NotImplementedError("File type not supported.")


@contextmanager
def cached(fn):
    """
    A context manager that caches the output of a computation to a file.

    Args:
        fn (str): The file path to which the cached data will be written.

    Yields:
        str: The file path of the cache file.

    Examples:

        with cached('a.txt') as cache_fn:
            with open(cache_fn, 'w') as w:
                w.write('123')

    """
    import shutil
    cache_fn = f'{fn}.lumo_cache'
    try:
        yield cache_fn
    except:
        os.remove(cache_fn)
    finally:
        shutil.move(cache_fn, fn)
