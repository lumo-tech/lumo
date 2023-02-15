import os
import sys
from functools import cache

from .config import LIBHOME, EXP_ROOT, CACHE_ROOT, BLOB_ROOT, METRIC_ROOT


def home():
    return os.path.expanduser("~")


@cache
def cache_dir():
    try:
        os.makedirs(CACHE_ROOT, exist_ok=True)
        res = CACHE_ROOT
    except PermissionError:
        res = os.path.join(home(), '.lumo/cache')
        os.makedirs(res, exist_ok=True)
    return res


@cache
def dataset_cache_dir(name=None):
    if name is None:
        res = os.path.join(cache_dir(), 'datasets', '__default__')
    else:
        res = os.path.join(cache_dir(), 'datasets', name)
    os.makedirs(res, exist_ok=True)
    return res


@cache
def libhome():
    if LIBHOME:
        return LIBHOME
    return os.path.join(home(), '.lumo')


@cache
def exproot():
    if EXP_ROOT:
        res = EXP_ROOT
    else:
        res = os.path.join(libhome(), 'experiments')
    os.makedirs(res, exist_ok=True)
    return res


@cache
def blobroot():
    if BLOB_ROOT:
        res = BLOB_ROOT
    else:
        res = os.path.join(libhome(), 'blob')
    os.makedirs(res, exist_ok=True)
    return res


@cache
def metricroot():
    if METRIC_ROOT:
        res = METRIC_ROOT
    else:
        res = os.path.join(libhome(), 'metrics')
    os.makedirs(res, exist_ok=True)
    return res


@cache
def local_dir():
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res is None:
        res = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
    return res
