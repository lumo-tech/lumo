"""
Safe means Exceptions won't be raised. When
 - call `dump_xxx` methods, True/False will be returned to indicate success/failed.
 - call `load_xxx` methods, None will be returned if something wrong happened.
"""
import json
import os
import textwrap
from collections import namedtuple
from importlib.machinery import SourceFileLoader

import torch

from lumo.decorators.safe import safe_dump, safe_load

pyval = namedtuple('pyval', ['name', 'value', 'annotation', 'commented'],
                   defaults=[None, None, False])


@safe_dump()
def dump_pyconfig(obj: list, fn: str):
    assert fn.endswith('.py')
    with open(fn, 'w', encoding='utf-8') as w:
        for (k, v, doc, commented) in obj:  # type:str
            if commented:
                commented = '# '
            else:
                commented = ''
            res = []
            if doc is not None:
                doc = str(doc)
                doc_ln = len(doc.split('\n'))
                if doc_ln > 1:
                    res.append(textwrap.indent(doc, '# '))
                    res.append('\n')

                if doc_ln == 0 or doc_ln > 1:
                    res.append(f'{commented}{k} = {v.__repr__()}\n')
                elif doc_ln == 1:
                    res.append(f'{commented}{k} = {v.__repr__()}  # {doc}\n')
            else:
                res.append(f'{commented}{k} = {v.__repr__()}\n')
            res.append('\n')
            w.write(''.join(res))
    return True


@safe_load()
def load_pyconfig(fn):
    module_name = os.path.splitext(os.path.basename(fn))[0]
    foo = SourceFileLoader(module_name, fn).load_module()
    res = {}
    for val_name in dir(foo):
        if val_name.startswith('__'):
            continue
        if isinstance(foo.__dict__[val_name], (list, dict, set, int, str, float)):
            res[val_name] = foo.__dict__[val_name]
    return res


@safe_dump()
def dump_json(obj, fn):
    with open(fn, 'w', encoding='utf-8') as w:
        json.dump(obj, w, indent=2)


@safe_dump()
def dump_yaml(obj, fn):
    import yaml
    with open(fn, 'w', encoding='utf-8') as w:
        yaml.safe_dump(obj, w)
    return fn


@safe_dump()
def dump_state_dict(obj, fn):
    torch.save(obj, fn)
    return fn


@safe_load(default={})
def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as r:
        return json.load(r)


@safe_load(default={})
def load_yaml(fn):
    import yaml
    with open(fn, 'r', encoding='utf-8') as r:
        return yaml.safe_load(r)


@safe_load()
def load_state_dict(fn: str, map_location='cpu'):
    ckpt = torch.load(fn, map_location=map_location)
    return ckpt


def remove_file(fn):
    if os.path.exists(fn) and os.path.isfile(fn):
        os.remove(fn)


@safe_load()
def load_string(fn):
    with open(fn, 'r', encoding='utf-8') as r:
        return ''.join(r.readlines())


@safe_dump()
def dump_string(string: str, fn, append=False):
    mode = 'w'
    if append:
        mode = 'a'
    with open(fn, mode, encoding='utf-8') as w:
        w.write(string)
    return fn


def safe_getattr(self, key, default=None):
    try:
        return getattr(self, key, default)
    except:
        return default
