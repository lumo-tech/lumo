import importlib

__all__ = ['get_lock']


def get_lock(*others):
    """
    Used to record the specific version of the run-time dependencies to ensure reproducibility.

    Args:
        *others: other library to be recorded.

    Returns:
        A dict instance with each library as the key and its version as the value.

    """
    res = {}

    for lib in others:
        mod = importlib.import_module(lib)
        res[lib] = getattr(mod, '__version__', 'null')

    return res
