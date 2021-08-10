"""
Methods about files/paths
"""
from lumo.utils import re
import os


def checkpath(*path, is_file=False):
    """

    Args:
        *path:

    Returns:

    """
    res = os.path.join(*path)
    if is_file:
        os.makedirs(os.path.dirname(res), exist_ok=True)
    else:
        os.makedirs(res, exist_ok=True)
    if not os.path.exists(res):
        return None
    return res


def compare_path(a, b):
    """
    Returns `True` if two path is equal, else `False`.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    a, b = os.path.realpath(a), os.path.realpath(b)
    return os.path.normpath(a) == os.path.normpath(b)


def walk_files(root, suf='.ft'):
    """
    Recursively yields all files ends with a given suffix under the `root` directory.
    File will be reformatted in absolute path.

    Args:
        root: iterate directory path.
        suf:

    Returns:

    """
    if suf != '':
        suf = f'{suf.lstrip(".")}'  # check suffix
    for root, dirs, fs in os.walk(root):
        for f in sorted(fs):
            if f.endswith(suf):
                yield os.path.join(root, f)


def filter_filename(title: str, substr='-'):
    """Create a valid path by replacing invalid character into the given `substr`."""
    title = re.sub('[\/:*?"<>|]', substr, title)
    return title
