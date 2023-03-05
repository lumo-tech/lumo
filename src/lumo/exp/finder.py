"""
list ->
    experiment
    test
retrieval ->
    test_name
    test_root

"""
from pprint import pformat
import os
from typing import List, Dict

from lumo.proc.path import libhome, exproot, metricroot
from lumo.utils.fmt import indent_print
from lumo.utils import re

from . import Experiment


def list_experiment_paths(exp_root=None):
    if exp_root is None:
        exp_root = exproot()
    return [os.path.join(exp_root, i) for i in os.listdir(exp_root)]


def _get_exp_name(exp_path: str):
    return os.path.basename(exp_path.rstrip('/'))


def list_all(exp_root=None) -> Dict[str, List[Experiment]]:
    return {
        _get_exp_name(exp_path): retrieval_tests_from_experiment(exp_path)
        for exp_path in list_experiment_paths(exp_root)
    }


def retrieval_tests_from_experiment(exp_path) -> List[Experiment]:
    return [retrieval_experiment(os.path.join(exp_path, f)) for f in os.listdir(exp_path)]


def list_test_names_from_experiment(experiment_name):
    return os.listdir(os.path.join(exproot(), experiment_name))


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
    """
    ^[0-9]{6}.[0-9]{3}.[a-z0-9]{2}t$
    """
    return re.search(r'^\d{6}\.\d{3}\.[a-z\d]{2}t$', test_name) is not None


def is_test_root(path: str):
    test_name = os.path.basename(path.rstrip('/'))
    return is_test_name(test_name)


def retrieval_test_root(test_flag: str):
    """
    test_flag can be a name like `230214.037.62t` or path like `path/to/230214.037.62t`
    """
    if is_test_name(test_flag):
        test_root = find_path_from_test_name(test_flag)
        if test_root is None:
            return None
    elif is_test_root(test_flag):
        test_root = test_flag
    else:
        return None

    return test_root


def retrieval_experiment(test_name=None, test_root: str = None):
    if test_root is None:
        test_root = retrieval_test_root(test_name)
    if test_root is None:
        return None
    exp = Experiment.from_disk(test_root)
    return exp


def summary_experiment(test_name: str = None, test_root: str = None):
    if test_root is None:
        if test_name is None:
            raise ValueError()
        test_root = retrieval_test_root(test_name)

    if test_root is None:
        return

    exp = retrieval_experiment(test_root)

    print('Properties:')
    indent_print(pformat(exp.properties))
    print('Tags:')
    indent_print(pformat(exp.tags))
    print('Use paths:')
    indent_print(pformat(exp.paths))
    print('Execute:')
    indent_print(' '.join(exp.exec_argv))
    print('-----------------------------------')


def format_experiment(exp: Experiment):
    return {
        'Properties': exp.properties,
        'tags': exp.tags,
        'paths': exp.paths,
        'exec_argv': exp.exec_argv,
    }


def list_all_metrics(metric_root=None):
    if metric_root is None:
        metric_root = metricroot()

    res = {}
    for root, dirs, fs in os.walk(metric_root):
        if root == metric_root:
            continue
        fs = [os.path.join(root, i) for i in fs if not i.startswith('.')]
        res[os.path.basename(root)] = fs
    return res
