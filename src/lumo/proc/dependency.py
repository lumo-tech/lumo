import importlib

from accelerate import __version__ as accelerate_version
from fire import __version__ as fire_version
from joblib import __version__ as joblib_version
from psutil import __version__ as psutil_version

from lumo import __version__ as lumo_version


class Version:
    # lumo = lumo_version
    joblib = joblib_version
    fire = fire_version
    psutil = psutil_version
    accelerate = accelerate_version


def get_lock(*others):
    res = {}
    res['lumo'] = lumo_version
    res.update({k: v for k, v in Version.__dict__.items() if not k.startswith('__')})

    for lib in others:
        mod = importlib.import_module(lib)
        res[lib] = getattr(mod, '__version__', 'null')

    return res
