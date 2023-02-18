import importlib

from accelerate import __version__ as accelerate_version
from fire import __version__ as fire_version
from joblib import __version__ as joblib_version
from psutil import __version__ as psutil_version

from lumo import __version__ as lumo_version

__all__ = ['get_lock']


class Version:
    # lumo = lumo_version
    joblib = joblib_version
    fire = fire_version
    psutil = psutil_version
    accelerate = accelerate_version


def get_lock(*others):
    """
    Used to record the specific version of the run-time dependencies to ensure reproducibility.

    Args:
        *others: other library to be recorded.

    Returns:
        A dict instance with each library as the key and its version as the value.

    """
    res = {}
    res['lumo'] = lumo_version
    res.update({k: v for k, v in Version.__dict__.items() if not k.startswith('__')})

    for lib in others:
        mod = importlib.import_module(lib)
        res[lib] = getattr(mod, '__version__', 'null')

    return res
