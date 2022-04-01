from typing import TypeVar, Callable

T = TypeVar('T', Callable, str)


def clswrap(callable: T) -> T:
    """
    带类型提示的 staticmethod()
    Args:
        callable:

    Returns:
    Notes:
        必须在类中用
    """

    @staticmethod
    def inner(*args, **kwargs):
        return callable(*args, **kwargs)

    return inner
