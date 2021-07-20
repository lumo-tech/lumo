"""
"""
import numpy as np
import pandas as pd
import torch
from numbers import Number
import hashlib
import os
import inspect
from typing import List, Tuple, Union, Mapping, Sequence, Set

from lumo.proc.path import cache_dir
from . import safe_io as io


def _to_string(item):
    res = item
    try:
        res = inspect.getsource(res)
        return res
    except TypeError:
        if inspect.isclass(res):
            res = res.__name__

    if isinstance(res, torch.Tensor):
        res = res.detach().cpu().numpy()

    if isinstance(res, np.ndarray):
        res = res.view(-1)
        mask = np.linspace(0, len(res), len(res) // 2, dtype=np.int)
        res = f"{res.shape}{res.dtype}{res[mask]}{res.sum()}{res.mean()}{res.std()}"

    if isinstance(res, pd.DataFrame):
        res = f"{res.columns}{res.shape}"

    if inspect.isclass(res) or inspect.isfunction(res):
        res = f"{res.__name__}"

    if not isinstance(res, (str, Number, Mapping, Sequence, Set)):
        res = f"{res.__class__.__name__}"
    else:
        res = f"{res}{res.__class__.__name__}"

    res = str(res)

    return res


def _cache_hash(*args, **kwargs) -> str:  # TODO hash any thing
    md5 = hashlib.md5()

    for arg in args:
        md5.update(_to_string(arg).encode(encoding='utf-8'))

    keys = sorted(list(kwargs.keys()))
    for k in keys:
        v = kwargs[k]
        md5.update(k.encode())
        md5.update(_to_string(v).encode(encoding='utf-8'))

    return md5.hexdigest()


def _cache_suffix(item):
    if isinstance(item, np.ndarray):
        return 'npy'
    elif isinstance(item, torch.Tensor):
        return 'pth'
    elif isinstance(item, pd.DataFrame):
        return 'ft'
    else:
        return 'pkl'


def _drop_pandas(item: pd.DataFrame):
    for key in {'index', 'level_0'}:
        if key in item.columns:
            item = item.drop(key, 1)
    return item


def _dump_cache(item, fn):
    if isinstance(item, pd.DataFrame):
        item = _drop_pandas(item)
        item = item.reset_index()
        item = _drop_pandas(item)
        try:
            item.to_feather(fn)
        except:
            item.to_pickle(fn)
    elif isinstance(item, np.ndarray):
        np.save(fn, item)
    elif isinstance(item, torch.Tensor):
        torch.save(item, fn)
    else:
        io.dump_state_dict(item, fn)


def _load_cache(fn: str):
    if fn.endswith('ft'):
        try:
            return pd.read_feather(fn)
        except:
            return pd.read_pickle(fn)
    elif fn.endswith('npy'):
        return np.load(fn)
    elif fn.endswith('pth'):
        return torch.load(fn, map_location='cpu')
    else:
        return io.load_state_dict(fn)


def save_cache(items: Union[List, Tuple], *args, **kwargs):
    hash = _cache_hash(*args, **kwargs)
    path = os.path.join(cache_dir(), hash)
    os.makedirs(path, exist_ok=True)
    for i, item in enumerate(items):
        cfn = f"cache_{i:03d}.{_cache_suffix(item)}"
        _dump_cache(item, os.path.join(path, cfn))
    return path


def load_from_cache_path(path):
    fs = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.startswith('cache')]
    items = [_load_cache(f) for f in fs]
    return items


def load_if_exists(*args, **kwargs):
    hash = _cache_hash(*args, **kwargs)
    path = os.path.join(cache_dir(), hash)
    if os.path.isdir(path):
        items = load_from_cache_path(path)
        return items, path
    return None, None
