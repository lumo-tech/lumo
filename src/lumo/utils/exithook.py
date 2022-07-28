"""
Do something you want before the programe exit.
"""
import sys
from functools import wraps


def replace(func):
    sys.excepthook = func


def wrap_after(func):
    old = sys.excepthook

    def outer(fun):
        @wraps(fun)
        def inner(*args, **kwargs):
            old(*args, **kwargs)
            fun(*args, **kwargs)

        return inner

    sys.excepthook = outer(func)


def wrap_before(func):
    old = sys.excepthook

    def outer(fun):
        @wraps(fun)
        def inner(*args, **kwargs):
            fun(*args, **kwargs)
            old(*args, **kwargs)

        return inner

    sys.excepthook = outer(func)
