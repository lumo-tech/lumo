import os
import sys
from functools import cache

from .config import LIBHOME, EXP_ROOT, CACHE_ROOT, BLOB_ROOT, METRIC_ROOT


def home():
    return os.path.expanduser("~")


@cache
def cache_dir():
    """
    Directory to store cache files, like datasets. Can be shared for everyone.

    Returns:
        An available path, default value is ~/.cache/.lumo,
        if the path cannot be used due to permission denied,
        `~/.lumo/cache` will be returned

    Notes:

    """
    try:
        os.makedirs(CACHE_ROOT, exist_ok=True)
        res = CACHE_ROOT
    except PermissionError:
        res = os.path.join(home(), '.lumo/cache')
        os.makedirs(res, exist_ok=True)
    return res


@cache
def libhome():
    """Library home to store configs. Default is `~/.lumo`"""
    if LIBHOME:
        return LIBHOME
    return os.path.join(home(), '.lumo')


@cache
def exproot():
    """Experiment root to store multiple experiments, default is `~/.lumo/experiments`"""
    if EXP_ROOT:
        res = EXP_ROOT
    else:
        res = os.path.join(libhome(), 'experiments')
    os.makedirs(res, exist_ok=True)
    return res


@cache
def blobroot():
    """Experiment root to store big files, default is `~/.lumo/blob`"""

    if BLOB_ROOT:
        res = BLOB_ROOT
    else:
        res = os.path.join(libhome(), 'blob')
    os.makedirs(res, exist_ok=True)
    return res


@cache
def metricroot():
    """
    Only used for storing table_row instance.

    See Also:
        `~lumo.core.disk.TableRow`

    """
    if METRIC_ROOT:
        res = METRIC_ROOT
    else:
        res = os.path.join(libhome(), 'metrics')
    os.makedirs(res, exist_ok=True)
    return res


@cache
def local_dir():
    """
    Project root, default is the parent directory of .git.
    If this program is not a git project, the parent directory of the executed file will be returned.

    Returns:

    """
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res is None:
        res = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
    return res
