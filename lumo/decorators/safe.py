import sys
import traceback


def safe_load(default=None, print_exc_stack=True, print_method_args=True):
    def outer(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                if print_method_args:
                    print(args, kwargs, file=sys.stderr)
                if print_exc_stack:
                    traceback.print_exc()
                return default

        return inner

    return outer


def safe_dump(print_exc_stack=True, print_method_args=True):
    def outer(func):
        def inner(*args, **kwargs):
            try:
                func(*args, **kwargs)
                return True
            except BaseException as e:
                if print_method_args:
                    print(args, kwargs, file=sys.stderr)
                if print_exc_stack:
                    traceback.print_exc()
                return False

        return inner

    return outer
