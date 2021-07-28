import os
import pickle as _pickle
from io import FileIO
from contextlib import contextmanager


def dump(obj, file, make_path=True, protocol=None, *, fix_imports=True):
    if isinstance(file, str):
        if make_path:
            os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
        file = open(file, 'wb')
        _pickle.dump(obj, file, protocol=protocol, fix_imports=fix_imports)
        file.close()
    elif isinstance(file, FileIO):
        _pickle.dump(obj, file, protocol=protocol, fix_imports=fix_imports)
    else:
        raise NotImplementedError()


def load(file, *, fix_imports=True, encoding="ASCII", errors="strict"):
    if isinstance(file, str):
        file = open(file, 'rb')
        res = _pickle.load(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
        file.close()
        return res
    elif isinstance(file, FileIO):
        return _pickle.load(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
    else:
        raise NotImplementedError()


def loads(s, *, fix_imports=True, encoding="ASCII", errors="strict"):
    return _pickle.loads(s, fix_imports=fix_imports, encoding=encoding, errors=errors)


def dumps(obj, protocol=None, *, fix_imports=True):
    return _pickle.dumps(obj, protocol=protocol, fix_imports=fix_imports)


@contextmanager
def cached(fn):
    import shutil
    cache_fn = f'{fn}.lumo_cache'
    try:
        yield cache_fn
    except:
        os.remove(cache_fn)
    finally:
        shutil.move(cache_fn, fn)
