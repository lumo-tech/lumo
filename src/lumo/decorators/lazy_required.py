from functools import wraps
from importlib import util as imputil

_lib_memory = set()


def is_lib_available(lib):
    if lib in _lib_memory:
        return True
    res = imputil.find_spec(lib)
    if res is None:
        return False

    _lib_memory.add(lib)
    return True


def torch_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_lib_available('torch'):
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires PyTorch.")

    return wrapper


def lib_required(lib_name):
    def outer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if is_lib_available(lib_name):
                return func(*args, **kwargs)
            else:
                raise ImportError(f"Method `{func.__name__}` requires {lib_name}.")

        return wrapper

    return outer
