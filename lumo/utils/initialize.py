import os

from lumo.utils import safe_io as io
from lumo.utils.hash import hash
from lumo.utils.keys import CFG


def _default_config():
    DEFAULT = CFG.PATH.DEFAULT
    return {
        CFG.PATH.GLOBAL_EXP: DEFAULT.GLOBAL_EXP,
        CFG.PATH.DATASET: DEFAULT.DATASET,
        CFG.PATH.PRETRAINS: DEFAULT.PRETRAINS,
        CFG.PATH.CACHE: DEFAULT.CACHE,
    }


def _generate_config(path):
    os.makedirs(path, exist_ok=True)
    configf = os.path.join(path, 'config.json')
    io.dump_json(_default_config(), configf)


def _generate_hash(path):
    hash_fn = os.path.join(path, '.hash')
    if os.path.exists(hash_fn):
        return

    with open(hash_fn, 'w', encoding='utf-8') as w:
        w.write(hash(hash_fn)[:4])


def _initialize_lumo_path(path, local):
    _generate_config(path)
    if local:
        _generate_hash(path)
