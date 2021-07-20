"""
Methods about files/paths/hash
"""
import os
from contextlib import contextmanager


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


@contextmanager
def cacheed(fn):
    import shutil
    cache_fn = f'{fn}.lumo_cache'
    try:
        yield cache_fn
    except:
        os.remove(cache_fn)
    finally:
        shutil.move(cache_fn, fn)
