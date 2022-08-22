from pprint import pformat
import os
from lumo.proc.path import libhome
from lumo.utils.fmt import indent_print

from . import Experiment


def find_experiments():
    return os.listdir(os.path.join(libhome(), 'experiment'))


def list_test_names_from_experiment(experiment):
    return os.listdir(os.path.join(libhome(), 'experiment', experiment))


def find_path_from_test_name(test_name: str):
    if not is_test_name(test_name):
        return None

    date, _, _ = test_name.split('.')
    f = os.path.join(libhome(), 'diary', f'{date}.log')
    if os.path.exists(f):
        with open(f) as r:
            for line in r:
                time, path = line.split(', ')
                if test_name in path:
                    return path.strip()
    return None


def is_test_name(test_name: str):
    chunk = test_name.split('.')
    if len(test_name) != 14:
        return False

    if len(chunk) != 3:
        return False
    for k, c in zip(chunk, [6, 3, 3]):
        if len(k) != c:
            return False
    for k in chunk[:2]:
        try:
            int(k)
        except:
            return False
    return True


def is_test_root(test_root: str):
    test_name = os.path.basename(test_root.rstrip('/'))
    return is_test_name(test_name)


def format_experiment(exp: Experiment):
    return {
        'Properties': exp.properties,
        'tags': exp.tags,
        'paths': exp.paths,
        'exec_argv': exp.exec_argv,
    }


def get_experiment_name(test_root:str):
    return os.path.basename(os.path.dirname(test_root.rstrip('/')))

def get_test_name(test_root:str):
    return os.path.basename(test_root.rstrip('/'))

def ensure_test_root(tid: str):
    if is_test_name(tid):
        test_root = find_path_from_test_name(tid)
        if test_root is None:
            return None
    elif is_test_root(tid):
        test_root = tid
    else:
        return None

    return test_root


def summary_experiment(tid: str):
    test_root = ensure_test_root(tid)
    if test_root is None:
        return test_root

    exp = Experiment.from_disk(test_root)

    print('Properties:')
    indent_print(pformat(exp.properties))
    print('Tags:')
    indent_print(pformat(exp.tags))
    print('Use paths:')
    indent_print(pformat(exp.paths))
    print('Execute:')
    indent_print(' '.join(exp.exec_argv))
    print('-----------------------------------')
