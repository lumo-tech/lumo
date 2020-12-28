import os
import textwrap
from textwrap import dedent
from importlib import util
import importlib
import inspect

from thexp import analyse, base_classes, calculate, contrib, decorators, frame, nest, utils

modules = [analyse, base_classes, calculate, contrib, decorators, frame, nest, utils]

cwd = os.getcwd()


def recursive():
    init_path = os.path.join(cwd, 'thexp')
    for root, dirs, fs in os.walk(init_path):
        if 'cli' in root or 'future' in root:
            continue
        for f in fs:
            pref, _ = os.path.splitext(f)
            file = os.path.join(root, pref)
            if '__' in file:
                continue
            m = os.path.relpath(file, cwd).replace('\\', '.')
            m = importlib.import_module(m)
            members = inspect.getmembers(m)
            for member in members:
                yield member

        # for d in dirs:
        #     file = os.path.join(root, d)
        #     if '__' in file:
        #         continue
        #     m = os.path.relpath(file, cwd).replace('\\', '.')
        #     m = importlib.import_module(m)
        #     members = inspect.getmembers(m)
        #     for member in members:
        #         yield member


def grab_md(module, attr):
    members = inspect.getmembers(attr)

    print("# {}".format(module))
    if attr.__doc__ is not None:
        print('## {}'.format(attr.__name__))
        print(dedent(attr.__doc__))
    if inspect.isclass(attr):
        for member in members:
            if inspect.isfunction(member[1]):
                if member[1].__doc__ is not None:
                    print('## {}'.format(member[1].__name__))
                    print(dedent(member[1].__doc__))
    elif inspect.isfunction(attr):
        # print(dedent(attr.__doc__))
        pass


api_root = 'thexp/doc/api'
os.makedirs(api_root, exist_ok=True)

module_names = [i.__name__.split('.') for i in modules]

mem = set()
members = recursive()
for member in members:
    try:
        module = member[1].__module__
        if 'thexp' in module:
            name = '{}.{}'.format(module, member[1].__name__)
            if name not in mem:
                if module.split('.')[:2] in module_names:
                    m = importlib.import_module(module)
                    attr = getattr(m, member[1].__name__)
                    grab_md(module, attr)
                    # print(m, attr)
                mem.add(name)

    except:
        # print(member[1])
        pass
