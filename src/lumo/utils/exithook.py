"""
This module provides functions to wrap or replace the default exception handler of Python's sys module.
"""
import sys
from functools import wraps


def replace(func):
    """Replace the default exception handler with the provided function."""
    sys.excepthook = func


def wrap_after(func):
    """Wrap the default exception handler, executing the provided function after it."""
    old = sys.excepthook

    def outer(fun):
        """wrap function"""
        @wraps(fun)
        def inner(*args, **kwargs):
            """wrap function"""
            old(*args, **kwargs)
            fun(*args, **kwargs)

        return inner

    sys.excepthook = outer(func)


def wrap_before(func):
    """Wrap the default exception handler, executing the provided function before it."""
    old = sys.excepthook

    def outer(fun):
        """wrap function"""
        @wraps(fun)
        def inner(*args, **kwargs):
            """wrap function"""
            fun(*args, **kwargs)
            old(*args, **kwargs)

        return inner

    sys.excepthook = outer(func)
