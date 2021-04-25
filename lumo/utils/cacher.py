"""
"""
import numpy as np
import pandas as pd
import torch
from numbers import Number
import hashlib
import os
import inspect
from typing import List, Tuple, Union

from .paths import cache_dir
from . import safe_io as io


def _to_string(item, encode=True):
    res = item
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

    if not isinstance(res, str):
        res = f"{res.__class__.__name__}"
    else:
        res = f"{res}{res.__class__.__name__}"

    res = str(res)
    if encode:
        res = res.encode()

    return res


def _cache_hash(func, *args, **kwargs) -> str:  # TODO hash any thing
    md5 = hashlib.md5()
    md5.update(func.__name__.encode())
    md5.update(func.__class__.__name__.encode())

    for arg in args:
        md5.update(_to_string(arg))

    keys = sorted(list(kwargs.keys()))
    for k in keys:
        v = kwargs[k]
        md5.update(k.encode())
        md5.update(_to_string(v))

    return md5.hexdigest()


def _cache_suffix(item):
    if isinstance(item, (list, tuple, dict, torch.Tensor)):
        return 'pkl'
    elif isinstance(item, np.ndarray):
        return 'npy'
    elif isinstance(item, pd.DataFrame):
        return 'ft'


def _drop_pandas(item: pd.DataFrame):
    for key in {'index', 'level_0'}:
        if key in item.columns:
            item = item.drop(key, 1)
    return item


def _dump_cache(item, fn):
    if isinstance(item, (list, tuple, dict, torch.Tensor, np.ndarray)):
        io.dump_state_dict(item, fn)
    elif isinstance(item, pd.DataFrame):
        item = _drop_pandas(item)
        item = item.reset_index()
        item = _drop_pandas(item)
        try:
            item.to_feather(fn)
        except:
            item.to_pickle(fn)


def _load_cache(fn: str):
    if fn.endswith('ft'):
        try:
            return pd.read_feather(fn)
        except:
            return pd.read_pickle(fn)
    else:
        return io.load_state_dict(fn)


def save_cache(items: Union[List, Tuple], func, *args, **kwargs):
    hash = _cache_hash(func, *args, **kwargs)
    path = os.path.join(cache_dir(), hash)
    os.makedirs(path, exist_ok=True)
    for i, item in enumerate(items):
        cfn = f"cache_{i:03d}.{_cache_suffix(item)}"
        _dump_cache(item, os.path.join(path, cfn))
    return path


def load_if_exists(func, *args, **kwargs):
    hash = _cache_hash(func, *args, **kwargs)
    path = os.path.join(cache_dir(), hash)
    if os.path.isdir(path):
        fs = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.startswith('cache')]
        items = [_load_cache(f) for f in fs]
        return items, path
    return None, None
