from typing import TypeVar, Callable

T = TypeVar('T', Callable, str)


def clswrap(callable: T) -> T:
    """
    A decorator for creating a staticmethod with type hints.

    Args:
        callable: The function to be wrapped with a staticmethod.

    Returns:
        A new staticmethod that calls the original function.

    Notes:
        This decorator should be used on a class method. It creates a new staticmethod that calls the original function,
        allowing it to be called without an instance of the class. The `callable` argument should have type hints
        for the parameters and return type. The resulting staticmethod will also have the same type hints.
    """

    @staticmethod
    def inner(*args, **kwargs):
        return callable(*args, **kwargs)

    return inner
