import os
import sys
from lumo.utils.cache import cache
from .config import glob


def home():
    """
    Returns the home directory of the current user.

    This function returns the home directory of the current user, which is
    different from the home directory of the library. If you need to access
    the library home directory, use `libhome()` instead.

    Returns:
        str: The home directory of the current user.
    """
    return os.path.expanduser("~")


def cache_dir():
    """
    Directory to store cache files, like datasets. Can be shared for everyone.

    Returns:
        An available path, default value is ~/.cache/.lumo,
        if the path cannot be used due to permission denied,
        `~/.lumo/cache` will be returned

    Notes:

    """
    CACHE_ROOT = glob.get('cache_dir', None)
    try:
        os.makedirs(CACHE_ROOT, exist_ok=True)
        res = CACHE_ROOT
    except PermissionError:
        res = os.path.join(home(), '.lumo/cache')
        os.makedirs(res, exist_ok=True)
    return res


def libhome():
    """
    Returns the library home directory.

    This function returns the library home directory, which is used to store
    configuration files. By default, the library home directory is `~/.lumo`.
    If a custom home directory is set in the `home` global variable, that
    directory will be returned instead.

    Returns:
        str: The library home directory.
    """
    LIBHOME = glob.get('home', None)
    if LIBHOME:
        return LIBHOME
    return os.path.join(home(), '.lumo')


def exproot():
    """Experiment root to store multiple experiments, default is `~/.lumo/experiments`"""
    EXP_ROOT = glob.get('exp_root', None)
    if EXP_ROOT:
        res = EXP_ROOT
    else:
        res = os.path.join(libhome(), 'experiments')

    os.makedirs(res, exist_ok=True)
    return res


def progressroot():
    """Experiment root to store multiple experiments, default is `~/.lumo/experiments`"""
    PROGRESS_ROOT = glob.get('progress_root', None)
    if PROGRESS_ROOT:
        res = PROGRESS_ROOT
    else:
        res = os.path.join(cache_dir(), 'progress')

    os.makedirs(res, exist_ok=True)
    return res


def blobroot():
    """Experiment root to store big files, default is `~/.lumo/blob`"""
    BLOB_ROOT = glob.get('blob_root', None)
    if BLOB_ROOT:
        res = BLOB_ROOT
    else:
        res = os.path.join(libhome(), 'blob')

    os.makedirs(res, exist_ok=True)
    return res


def metricroot():
    """
    Only used for storing table_row instance.

    See Also:
        `~lumo.core.disk.TableRow`

    """
    METRIC_ROOT = glob.get('metric_root', None)
    if METRIC_ROOT:
        res = METRIC_ROOT
    else:
        res = os.path.join(libhome(), 'metrics')
    os.makedirs(res, exist_ok=True)
    return res


def dbroot():
    """Root path to store experiment information. Default is `~/.lumo/database`"""
    DB_ROOT = glob.get('db_root', None)
    if DB_ROOT:
        res = DB_ROOT
    else:
        res = os.path.join(libhome(), 'database')
    os.makedirs(res, exist_ok=True)
    return res


def local_dir():
    """
    Project root, default is the parent directory of .git.
    If this program is not a git project, the parent directory of the executed file will be returned.

    Returns:

    """
    from lumo.utils.repository import git_dir
    res = git_dir()
    if res is None:
        res = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
    return res
