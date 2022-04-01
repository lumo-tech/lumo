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
    with open(fn, 'w', encoding='utf-8') as w:
        json.dump(obj, w, indent=2)


def dump_yaml(obj, fn):
    import yaml
    with open(fn, 'w', encoding='utf-8') as w:
        yaml.safe_dump(obj, w)
    return fn


def dump_state_dict(obj, fn):
    torch.save(obj, fn)
    return fn


def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as r:
        return json.load(r)


def load_yaml(fn):
    import yaml
    with open(fn, 'r', encoding='utf-8') as r:
        return yaml.safe_load(r)


def load_state_dict(fn: str, map_location='cpu'):
    ckpt = torch.load(fn, map_location=map_location)
    return ckpt


def load_text(fn):
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
    import shutil
    cache_fn = f'{fn}.lumo_cache'
    try:
        yield cache_fn
    except:
        os.remove(cache_fn)
    finally:
        shutil.move(cache_fn, fn)
