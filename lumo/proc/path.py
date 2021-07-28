import os
import sys


def cache_dir():
    res = os.path.expanduser("~/.cache/lumo")
    os.makedirs(res, exist_ok=True)
    return res


def dataset_cache_dir(name=None):
    if name is None:
        return os.path.join(cache_dir(), 'datasets')
    else:
        return os.path.join(cache_dir(), 'datasets', name)


def libhome():
    return os.path.expanduser("~/.lumo")


def git_dir(root='./'):
    """
    git repository directory
    Args:
        root:

    Returns:

    """
    from lumo.proc.explore import git_enable
    if git_enable():
        from git import Git
        cur = os.getcwd()
        os.chdir(root)
        res = Git().execute(['git', 'rev-parse', '--git-dir'])
        res = os.path.abspath(os.path.dirname(res))
        os.chdir(cur)
        return res
    else:
        return None


def local_dir():
    res = git_dir()
    if res is None:
        res = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
    return res


def global_config_path():
    return os.path.join(libhome(), "config.json")
