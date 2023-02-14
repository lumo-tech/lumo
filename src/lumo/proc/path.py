import os
import sys
from .config import LIBHOME, EXP_ROOT, CACHE_ROOT, BLOB_ROOT


def home():
    return os.path.expanduser("~")


def cache_dir():
    try:
        os.makedirs(CACHE_ROOT, exist_ok=True)
        res = CACHE_ROOT
    except PermissionError:
        res = os.path.join(home(), '.lumo/.cache')
        os.makedirs(res, exist_ok=True)
    return res


def dataset_cache_dir(name=None):
    if name is None:
        res = os.path.join(cache_dir(), 'datasets', '__default__')
    else:
        res = os.path.join(cache_dir(), 'datasets', name)
    os.makedirs(res, exist_ok=True)
    return res


def libhome():
    if LIBHOME:
        return LIBHOME
    return os.path.join(home(), '.lumo')


def exproot():
    if EXP_ROOT:
        return EXP_ROOT
    return os.path.join(libhome(), 'experiments')


def blobroot():
    if BLOB_ROOT:
        return BLOB_ROOT
    return os.path.join(libhome(), 'blob')


def local_dir():
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res is None:
        res = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
    return res
