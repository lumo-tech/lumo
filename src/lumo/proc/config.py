import json
import os

__all__ = ['debug_mode', 'glob', 'global_config_path', 'local_config_path']

import tempfile

GLOBAL_DEFAULT = {
    'home': os.path.expanduser("~/.lumo/"),
    'cache_dir': os.path.expanduser("~/.cache/lumo"),
    'dev_branch': 'lumo_experiments',
}


def global_config_path():
    """
    Returns the path to the global configuration file.

    Returns:
        str: The path to the global configuration file.

    Notes:
        The relative path of global config path should never change (~/.lumorc.json)
    """
    return os.path.expanduser("~/.lumorc.json")


def local_config_path():
    """
    Returns the path to the local configuration file.

    Returns:
        str: The path to the local configuration file, if found. Otherwise, None.
    """
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res:
        return os.path.join(res, ".lumorc.json")
    return None


def local_public_config_path():
    """
    Returns the path to the local configuration file that can be shared and public.

    Returns:
        str: The path to the local configuration file, if found. Otherwise, None.
    """
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res:
        return os.path.join(res, ".lumorc.public.json")
    return None


def get_config(path, default):
    """
    Reads the configuration file at the given path or creates it if it doesn't exist.

    Args:
        path (str): The path to the configuration file.
        default (dict): The default configuration to use if the file doesn't exist.

    Returns:
        dict: The configuration read from the file or the default configuration if the file doesn't exist.

    Raises:
        Exception: If there was an error reading the configuration file.
    """
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
    """
    Returns the runtime configuration by merging the global and local configurations.

    Returns:
        dict: The merged runtime configuration.
    """
    # default
    cfg = GLOBAL_DEFAULT

    # global config (~/.lumorc.json)
    glob_cfg = get_config(global_config_path(), GLOBAL_DEFAULT)
    cfg.update(glob_cfg)

    # local private config ({repo}/.lumorc.json)
    local_cfg = get_config(local_config_path(), {})
    cfg.update(local_cfg)

    # local public config ({repo}/.lumorc.public.json)
    local_public_cfg = get_config(local_public_config_path(), {})
    cfg.update(local_public_cfg)
    return cfg


def debug_mode(base_dir=None, disable_git=True):
    """Sets up global variables for debugging mode.

    Args:
        base_dir (str, optional): The directory to create temporary directories in. Defaults to None.
        disable_git (bool, optional): Whether to disable git hooks. Defaults to True.

    Returns:
        None
    """
    glob['exp_root'] = tempfile.mkdtemp(dir=base_dir)
    glob['db_root'] = tempfile.mkdtemp(dir=base_dir)
    glob['progress_root'] = tempfile.mkdtemp(dir=base_dir)
    glob['metric_root'] = tempfile.mkdtemp(dir=base_dir)

    glob['home'] = tempfile.mkdtemp(dir=base_dir)
    glob['cache_dir'] = tempfile.mkdtemp(dir=base_dir)
    glob['blob_root'] = tempfile.mkdtemp(dir=base_dir)
    # glob['HOOK_LOCKFILE'] = False
    glob['HOOK_LASTCMD_DIR'] = tempfile.mkdtemp(dir=base_dir)
    # glob['HOOK_RECORDABORT'] = False
    glob['HOOK_TIMEMONITOR'] = False

    if disable_git:
        glob['HOOK_GITCOMMIT'] = False


# A dict object contains runtime configuration.
glob = get_runtime_config()
