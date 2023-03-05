import json
import os

__all__ = ['debug_mode', 'glob', 'global_config_path', 'local_config_path']

import tempfile
from typing import overload

GLOBAL_DEFAULT = {
    'home': os.path.expanduser("~/.lumo/"),
    'cache_dir': os.path.expanduser("~/.cache/lumo"),
    'dev_branch': 'lumo_experiments',
}


def global_config_path():
    return os.path.expanduser("~/.lumorc.json")


def local_config_path():
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res:
        return os.path.join(res, ".lumorc.json")
    return None


def get_config(path, default):
    if path is None:
        return default

    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as r:
                config = json.load(r)
            return config
        except Exception as e:
            print(f'Error read {path}')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as w:
        json.dump(default, w)
    return default


def get_runtime_config():
    glob_cfg = get_config(global_config_path(), GLOBAL_DEFAULT)
    local_cfg = get_config(local_config_path(), {})
    cfg = GLOBAL_DEFAULT
    cfg.update(glob_cfg)
    cfg.update(local_cfg)
    return cfg


def debug_mode(base_dir=None, disable_git=True):
    glob['exp_root'] = tempfile.mkdtemp(dir=base_dir)
    glob['progress_root'] = tempfile.mkdtemp(dir=base_dir)
    glob['home'] = tempfile.mkdtemp(dir=base_dir)
    glob['cache_dir'] = tempfile.mkdtemp(dir=base_dir)
    glob['blob_root'] = tempfile.mkdtemp(dir=base_dir)
    glob['metric_root'] = tempfile.mkdtemp(dir=base_dir)
    glob['HOOK_LOCKFILE'] = False
    glob['HOOK_LASTCMD_DIR'] = tempfile.mkdtemp(dir=base_dir)
    glob['HOOK_RECORDABORT'] = False
    glob['HOOK_TIMEMONITOR'] = False

    if disable_git:
        glob['HOOK_GITCOMMIT'] = False


# A dict object contains runtime configuration.
glob = get_runtime_config()
