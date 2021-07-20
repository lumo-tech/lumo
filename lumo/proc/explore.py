from functools import lru_cache
from .const import ENV


@lru_cache(1)
def git_enable():
    if not ENV.IS_GIT_ENABLED:
        return False

    try:
        import git
    except ImportError:
        return False
    try:
        git.Git().execute(['git', 'rev-parse', '--git-dir'])
        return True
    except git.GitCommandError:
        return False


@lru_cache(1)
def matplotlib_enable():
    try:
        import matplotlib
        return True
    except ImportError:
        return False


@lru_cache(1)
def numpy_enable():
    try:
        import numpy
        return True
    except ImportError:
        return False


@lru_cache(1)
def pandas_enable():
    try:
        import pandas
        return True
    except ImportError:
        return False


@lru_cache(1)
def accelerate_enable():
    try:
        import accelerate
        return True
    except ImportError:
        return False
