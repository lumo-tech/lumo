"""
Format date/filename, check array shape, convert item from torch.Tensor to ndarray or scalar.
"""
import textwrap
from datetime import datetime

import numpy as np
import torch

from . import re


def to_ndarray(item):
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu()
    return np.array(item)


def detach(item):
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu().numpy()
    return item


def validate_scalar_shape(ndarray, name=''):
    if ndarray.ndim != 0:
        raise ValueError(
            "Expected scalar value for %r but got %r" % (name, ndarray)
        )
    return ndarray


def is_scalar(ndarray: np.ndarray):
    return ndarray.size == 1


def strftime(fmt='%y-%m-%d-%H%M%S', dateobj: datetime = None):
    """get current date with formatted"""
    if dateobj is not None:
        return dateobj.strftime(fmt)
    return datetime.now().strftime(fmt)


_invalid_fc = (
    r"[+?@#$&%*()=;|,<>: +"
    r"\^\-\/\t\b\[\]\"]+"
)


def to_filename(basename):
    return re.sub(_invalid_fc, '_', basename)


def can_be_filename(basename):
    return re.search(_invalid_fc, basename) is None


def indent_print(text, indent='    '):
    print(textwrap.indent(text, indent))
