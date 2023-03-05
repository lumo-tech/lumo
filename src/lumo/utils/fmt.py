"""
Format date/filename, check array shape, convert item from torch.Tensor to ndarray or scalar.
"""
import textwrap
from datetime import datetime

import numpy as np
import torch

from . import re


def to_ndarray(item):
    """Convert a PyTorch tensor or any other array-like object to a NumPy ndarray."""
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu()
    return np.array(item)


def detach(item):
    """Detach a PyTorch tensor and convert it to a NumPy ndarray."""
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu().numpy()
    return item


def validate_scalar_shape(ndarray, name=''):
    """Validate that a given numpy array is a scalar."""
    if ndarray.ndim != 0:
        raise ValueError(
            "Expected scalar value for %r but got %r" % (name, ndarray)
        )
    return ndarray


def is_scalar(ndarray: np.ndarray):
    """Check whether a numpy array is a scalar."""
    return ndarray.size == 1


def strftime(fmt='%y-%m-%d-%H%M%S', dateobj: datetime = None):
    """get current date with formatted"""
    if dateobj is not None:
        return dateobj.strftime(fmt)
    return datetime.now().strftime(fmt)


def strptime(fmt='%y-%m-%d-%H%M%S', datestr: str = None):
    """Convert a string to a datetime object using the specified format."""
    return datetime.strptime(datestr, fmt)


_invalid_fc = (
    r"[+?@#$&%*()=;|,<>: +"
    r"\^\-\/\t\b\[\]\"]+"
)


def to_filename(basename):
    """Replace invalid characters in a basename with an underscore."""
    return re.sub(_invalid_fc, '_', basename)


def can_be_filename(basename):
    """Check whether a basename can be converted to a valid filename."""
    return re.search(_invalid_fc, basename) is None


def indent_print(text, indent='    '):
    """Prints the specified text with a given indentation."""
    print(textwrap.indent(text, indent))


def format_second(sec: int) -> str:
    """Formats a duration given in seconds into a human-readable string."""
    sec, ms = divmod(sec, 1)
    if sec > 60:
        min, sec = divmod(sec, 60)
        if min > 60:
            hour, min = divmod(min, 60)
            fmt = "{}h{}m{}s".format(hour, min, int(sec))
        else:
            fmt = "{}m{}s".format(min, int(sec))
    else:
        fmt = "{}s".format(int(sec))
    return fmt
