"""
Methods about files/paths/hash
"""
import os
from functools import lru_cache

from git import Git

from .initialize import _initialize_lumo_path


def home_dir():
    """dir for storing global record"""
    path = os.path.expanduser("~/.lumo")
    if not os.path.exists(path):
        _initialize_lumo_path(path, local=False)
    return path


def local_dir(repo_=None):
    if repo_ is None:
        repo_ = os.getcwd()
    path = repo_dir(repo_)
    if path is None:
        return None
    path = os.path.join(path, '.lumo')
    if not os.path.exists(path):
        _initialize_lumo_path(path, local=True)
    return path


@lru_cache()
def repo_dir(root="./", ignore_info=False):
    """
    判断某目录是否在git repo 目录内（包括子目录），如果是，返回该 repo 的根目录
    :param root:  要判断的目录。默认为程序运行目录
    :return:

    如果是，返回该repo的根目录（包含 .git/ 的目录）
        否则，返回空
    """
    cur = os.getcwd()
    os.chdir(root)
    try:
        res = Git().execute(['git', 'rev-parse', '--git-dir'])

        res = os.path.abspath(os.path.dirname(res))
    except Exception as e:
        if not ignore_info:
            print(e)
        res = None
    os.chdir(cur)
    return res


def global_config_path():
    return os.path.join(home_dir(), "config.json")


def local_config_path(repo_=None):
    return os.path.join(local_dir(repo_), 'config.json')


def checkpath(*path):
    res = os.path.join(*path)
    os.makedirs(res, exist_ok=True)
    if not os.path.exists(res):
        return None
    return res


def compare_path(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    a, b = os.path.realpath(a), os.path.realpath(b)
    return os.path.normpath(a) == os.path.normpath(b)


def cache_dir():
    path = os.path.expanduser("~/.cache/lumo/")
    os.makedirs(path, exist_ok=True)
    return path
