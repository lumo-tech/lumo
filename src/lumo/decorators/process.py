from typing import Callable
from functools import wraps
from lumo.proc.dist import is_main


def call_on_main_process_wrap(func) -> Callable:
    """
    Wrap a function to only execute on the main process.

    If the current process is the main process, it calls the original func with the passed arguments and returns the result.
    If it is not the main process, it does nothing and returns None.

    Args:
        func (callable): The function to wrap.

    Returns:
        callable: A wrapped function that only executes on the main process.

    """

    @wraps(func)
    def inner(*args, **kwargs):
        if is_main():
            return func(*args, **kwargs)

    return inner
