import os
import sys

from lumo.utils.repository import git_dir


def cache_dir():
    try:
        res = os.path.expanduser("~/.cache/lumo")
        os.makedirs(res, exist_ok=True)
    except PermissionError:
        res = os.path.expanduser("~/.lumo/.cache")
        os.makedirs(res, exist_ok=True)
    return res


def dataset_cache_dir(name=None):
    if name is None:
        res = os.path.join(cache_dir(), 'datasets')
    else:
        res = os.path.join(cache_dir(), 'datasets', name)
    os.makedirs(res, exist_ok=True)
    return res


def libhome():
    return os.path.expanduser("~/.lumo")


def local_dir():
    res = git_dir()
    if res is None:
        res = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
    return res


def global_config_path():
    return os.path.join(libhome(), "config.json")
