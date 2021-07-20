import sys
import textwrap
import traceback


def safe_load(default=None, print_exc_stack=True, print_method_args=True):
    def outer(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                traceback.print_exc()
                print('=' * 80, file=sys.stderr)
                size = 80
                print('| ignored exception from safe_load()', file=sys.stderr)
                if print_method_args:
                    arg_info = f"| {args, kwargs}"
                    print(arg_info, file=sys.stderr)
                    size = len(arg_info)
                if print_exc_stack:
                    exc_info = traceback.format_exc().strip()
                    size = max(len(i) for i in exc_info.split('\n')) + 10
                    print(textwrap.indent(exc_info, '| '), file=sys.stderr)
                print('=' * size, file=sys.stderr)
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
                print('=' * 20, file=sys.stderr)
                if print_method_args:
                    print(args, kwargs, file=sys.stderr)
                if print_exc_stack:
                    traceback.print_exc()
                print('=' * 20, file=sys.stderr)
                return False

        return inner

    return outer
