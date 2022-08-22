import os
import json


def global_config_path():
    return os.path.expanduser("~/.lumorc.json")


def local_config_path():
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res:
        return os.path.join(res, ".lumorc.json")
    return None


def get_config(path):
    if path is None:
        return {}
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

HOME = glob.get("home")
CACHEROOT = glob.get('cache_dir')
EXPROOT = glob.get('exproot')
