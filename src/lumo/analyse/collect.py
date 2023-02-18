import os.path
import shutil
from typing import Iterable

import pandas as pd
from dbrecord import PDict

from lumo import Logger
from lumo.exp.finder import list_all_metrics
from lumo.proc.path import metricroot
from lumo.utils import safe_io as IO


def recursive_get(dic: dict, key: str, default=None):
    """
    Recursively retrieval value with provided key.
    The keys include hierarchical information should be separated by dot, like "a.b"

    Args:
        dic: a dict instance
        key: string
        default: default value if provided key not exists.

    Returns:
        `value` in the `dic`

    Caution:
        The `key` argument will be splited by '.' at first, so any value stored like "a.b" won't be retrieved.

    Examples:
        dic = {"a":{"b":1},"c":2}
        assert recursive_get(dic,'a.b') == 1
        assert recursive_get(dic,'c') == 2
    """
    for key in key.split('.'):
        if isinstance(dic, dict):
            dic = dic.get(key, default)
        else:
            return dic

    return dic


def flatten_dict(df, key: str, keys: Iterable, prefix: str = None, default=None):
    """
    The behaviour of
    `flatten_dict(df, "a",["b"])` are the same as
    ```
    for key in ["b"]:
        df[f"a.{key}"] = df["a"].apply(lambda x:x.get(key))
    ```
    """
    if prefix is None:
        prefix = key
    for k in keys:
        df[f'{prefix}.{k}'] = df[key].apply(lambda x: recursive_get(x, k, default))


def flatten_params(df, *keys: str):
    """See flatten_dict for details"""
    return flatten_dict(df, 'params', keys, prefix='p')


def flatten_metric(df, *keys: str):
    """See flatten_dict for details"""
    return flatten_dict(df, 'metric', keys, prefix='m')


def collect_table_rows(metric_root=None) -> pd.DataFrame:
    """Collect all table_row into a pandas.DataFrame"""
    res = []
    logger = Logger()
    exp_map = list_all_metrics(metric_root)
    for k, rows in exp_map.items():
        # append existing row metrics
        global_dic = PDict(os.path.join(metricroot(), f'{k}.dict.sqlite'))
        for row in global_dic.values():
            res.append(row)

        if len(rows) == 0:
            continue

        logger.info(f'collecting {len(rows)} tests.')

        for row_fn in rows:
            if not row_fn.endswith('pkl'):
                continue
            test_name = os.path.splitext(os.path.basename(row_fn))[0]
            try:
                row = IO.load_pkl(row_fn)
            except:
                print(f'Failed on load {row_fn}, renameed to f ".{test_name}.failed.pkl"')
                shutil.move(row_fn, os.path.join(os.path.dirname(row_fn), f'.{test_name}.failed.pkl'))
                continue
            global_dic[test_name] = row
            shutil.move(row_fn, os.path.join(os.path.dirname(row_fn), f'.{test_name}.pkl'))
            res.append(row)
        global_dic.flush()

    return pd.DataFrame(res)


def replac(df: pd.DataFrame):
    """replace after filtering"""
    raise NotImplementedError()
