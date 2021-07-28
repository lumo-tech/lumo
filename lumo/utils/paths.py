"""
Methods about files/paths/hash
"""
import os


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


def walk_files(root, suf='.ft'):
    if suf != '':
        suf = f'{suf.lstrip(".")}'  # check suffix
    for root, dirs, fs in os.walk(root):
        for f in sorted(fs):
            if f.endswith(suf):
                yield os.path.join(root, f)
