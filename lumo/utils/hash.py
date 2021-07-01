import hashlib
import os
from itertools import chain

from torch import nn

from . import re


def file_atime_hash(file):
    """
    calculate hash of given file's atime
    atime : time of last access
    """
    return string_hash(str(os.path.getatime(file)))


def string_hash(*str):
    """calculate hash of given string list"""
    hl = hashlib.md5()
    for s in str:
        hl.update(s.encode(encoding='utf-8'))
    return hl.hexdigest()[:16]


def file_hash(file: str) -> str:
    """calculate hash of given file path"""
    hl = hashlib.md5()
    with open(file, 'rb') as r:
        hl.update(r.read())
    return hl.hexdigest()[:16]


def filter_filename(title: str, substr='-'):
    """replace invalid string of given file path by `substr`"""
    title = re.sub('[\/:*?"<>|]', substr, title)  # 去掉非法字符
    return title


def inthash(value) -> int:
    return int(hash(value), 16)


def hash(value, hexint=False) -> str:
    """try to calculate hash of any given object"""
    import hashlib
    from collections.abc import Iterable
    from numbers import Number
    import numpy as np
    from torch import Tensor
    hl = hashlib.md5()

    if isinstance(value, (np.ndarray, Tensor)):
        if isinstance(value, Tensor):
            value = value.detach_().cpu().numpy()
        try:
            value = value.item()
        except ValueError:  # not an one element tensor
            value = str(value)
    if isinstance(value, (Number)):
        value = str(value)

    if isinstance(value, dict):
        for k in sorted(value.keys()):
            v = value[k]
            hl.update(str(k).encode(encoding='utf-8'))
            hl.update(hash(v).encode(encoding='utf-8'))
    elif isinstance(value, str):
        hl.update(value.encode(encoding='utf-8'))
    elif isinstance(value, Iterable):
        for v in value:
            hl.update(hash(v).encode(encoding='utf-8'))

    res = hl.hexdigest()
    if hexint:
        res = str(int(res, 16))
    return res


def hash_model(model: nn.Module):
    """
    If two model have equal model hash, then their paramters can be shared with each other.
    Commonly used for loading pretrained model.
    """
    hl = hashlib.md5()

    named_iter = chain(
        model.named_modules(),
        model.named_parameters(),
        model.named_children(),
        model.named_buffers(),
    )
    for k, _ in named_iter:  # type:str
        hl.update(k.encode(encoding='utf-8'))

    return hl.hexdigest()
