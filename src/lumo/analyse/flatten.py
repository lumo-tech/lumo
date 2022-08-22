from typing import Iterable


def recursive_get(dic: dict, key: str, default=None):
    for key in key.split('.'):
        if isinstance(dic, dict):
            dic = dic.get(key, default)
        else:
            return dic

    return dic


def flatten_dict(df, key: str, keys: Iterable, prefix: str = None, default=None):
    if prefix is None:
        prefix = key
    for k in keys:
        df[f'{prefix}.{k}'] = df[key].apply(lambda x: recursive_get(x, k, default))


def flatten_params(df, *keys: str):
    return flatten_dict(df, 'params', keys, prefix='p')


def flatten_metric(df, *keys: str):
    return flatten_dict(df, 'metric', keys, prefix='m')
