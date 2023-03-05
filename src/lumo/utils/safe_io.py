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


def dump_json(obj, fn):
    """
    Dumps the given object to a JSON file at the given file path.

    Args:
        obj: The object to be dumped to JSON.
        fn (str): The file path to which the JSON data will be written.

    Notes:
        The JSON data will be written with an indentation of 2 spaces.
    """
    with open(fn, 'w', encoding='utf-8') as w:
        json.dump(obj, w, indent=2)


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
    torch.save(obj, fn)


def load_json(fn):
    """Loads JSON data from the given file path and returns the resulting object."""
    with open(fn, 'r', encoding='utf-8') as r:
        return json.load(r)


def load_yaml(fn):
    """Loads YAML data from the given file path and returns the resulting object."""
    import yaml
    with open(fn, 'r', encoding='utf-8') as r:
        return yaml.safe_load(r)


def load_state_dict(fn: str, map_location='cpu'):
    ckpt = torch.load(fn, map_location=map_location)
    return ckpt


def load_text(fn):
    """Loads text data from the given file path and returns it as a single string."""
    if not os.path.exists(fn):
        return ''
    with open(fn, 'r', encoding='utf-8') as r:
        return ''.join(r.readlines())


def dump_text(string: str, fn, append=False):
    mode = 'w'
    if append:
        mode = 'a'
    with open(fn, mode, encoding='utf-8') as w:
        w.write(string)
    return fn


def safe_getattr(self, key, default=None):
    try:
        return getattr(self, key, default)
    except:
        return default


def dump_pkl(obj, file, make_path=True, protocol=None, *, fix_imports=True):
    if isinstance(file, str):
        if make_path:
            os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
        file = open(file, 'wb')
        _pickle.dump(obj, file, protocol=protocol, fix_imports=fix_imports)
        file.close()
    elif isinstance(file, FileIO):
        _pickle.dump(obj, file, protocol=protocol, fix_imports=fix_imports)
    else:
        raise NotImplementedError()


def load_pkl(file, *, fix_imports=True, encoding="ASCII", errors="strict"):
    if isinstance(file, str):
        file = open(file, 'rb')
        res = _pickle.load(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
        file.close()
        return res
    elif isinstance(file, FileIO):
        return _pickle.load(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
    else:
        raise NotImplementedError()


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
