import json
import os
import sys


def cache_dir():
    res = glob.get('libhome', None)
    if res is None:
        try:
            res = os.path.expanduser("~/.cache/lumo")
            os.makedirs(res, exist_ok=True)
        except PermissionError:
            res = os.path.expanduser("~/.lumo/.cache")
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
    return os.path.expanduser("~/.lumo")
    # lib_home = glob.get('libhome', None)
    # if lib_home is None:
    # return lib_home


def exproot():
    exp_root = glob.get('exproot', None)
    if exp_root is None:
        exp_root = libhome()
    return exp_root


def local_dir():
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res is None:
        res = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
    return res


def global_config_path():
    return os.path.join(libhome(), "config.json")


def local_config_path():
    res = local_dir()
    return os.path.join(res, "config.json")


def get_config(path):
    if os.path.exists(path):
        try:
            with open(path) as r:
                config = json.load(r)
            return config
        except Exception as e:
            print(f'Error read {path}')

    return {}


def create_runtime_config():
    glob_cfg = get_config(global_config_path())
    local_cfg = get_config(local_config_path())
    cfg = {}
    cfg.update(glob_cfg)
    cfg.update(local_cfg)
    return cfg


def pretain_model_path(name):
    pass


glob = create_runtime_config()
