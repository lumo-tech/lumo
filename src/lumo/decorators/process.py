from typing import Callable

from lumo.proc.dist import is_main


def call_on_main_process_wrap(func) -> Callable:
    def inner(*args, **kwargs):
        if is_main():
            return func(*args, **kwargs)

    return inner
