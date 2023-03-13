from functools import wraps
from importlib import util as imputil

_lib_memory = set()


def is_lib_available(lib):
    """
    Check if a library is available to be imported.

    Args:
        lib (str): The name of the library to check for.

    Returns:
        bool: True if the library is available, False otherwise.

    """
    if lib in _lib_memory:
        return True
    res = imputil.find_spec(lib)
    if res is None:
        return False

    _lib_memory.add(lib)
    return True


def torch_required(func):
    """
    Wrap a function to raise an ImportError if PyTorch is not available.
    """

    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_lib_available('torch'):
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires PyTorch.")

    return wrapper


def lib_required(lib_name):
    """
    Wrap a function to raise an ImportError if a required library is not available.
    """

    def outer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if is_lib_available(lib_name):
                return func(*args, **kwargs)
            else:
                raise ImportError(f"Method `{func.__name__}` requires {lib_name}.")

        return wrapper

    return outer
