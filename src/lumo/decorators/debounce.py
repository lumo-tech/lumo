import time
from functools import wraps


def debounce(wait):
    """
    debounce

    Args:
        wait (float): seconds

    Returns:
        function:

    Examples:
        @debounce(5)
        def my_func():
            print("my_func called")

        while True:
            my_func()
            time.sleep(1)
    """

    def decorator(func):
        last_time = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_time
            now = time.time()
            if now - last_time > wait:
                last_time = now
                return func(*args, **kwargs)

        return wrapper

    return decorator
