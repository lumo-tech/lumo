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
from typing import List, Dict, Any

from lumo.exp.watch import is_test_root, is_test_name
from lumo.proc.path import libhome, exproot, metricroot
from lumo.utils.fmt import indent_print

from . import Experiment


def list_experiment_paths(exp_root=None) -> List[str]:
    """
    Returns a list of experiment paths under exp_root directory.

    Args:
        exp_root: The root directory to search for experiments. Default is None, which uses the default experiment root directory.

    Returns:
        A list of experiment paths.
    """
    if exp_root is None:
        exp_root = exproot()
    return [os.path.join(exp_root, i) for i in os.listdir(exp_root)]


def _get_exp_name(exp_path: str) -> str:
    """
    Returns the name of the experiment directory.

    Args:
        exp_path: The path to the experiment directory.

    Returns:
        The name of the experiment directory.
    """
    return os.path.basename(exp_path.rstrip('/'))


def list_all(exp_root=None) -> Dict[str, List[Experiment]]:
    """
    Returns a dictionary of all experiments under exp_root directory.

    Args:
        exp_root: The root directory to search for experiments. Default is None, which uses the default experiment root directory.

    Returns:
        A dictionary of all experiments, where the keys are the names of the experiments and the values are lists of corresponding Experiment objects.
    """
    return {
        _get_exp_name(exp_path): retrieval_tests_from_experiment(exp_path)
        for exp_path in list_experiment_paths(exp_root)
    }


def retrieval_tests_from_experiment(exp_path: str) -> List[Experiment]:
    """
    Returns a list of Experiment objects found in the specified experiment directory.

    Args:
        exp_path: The path to the experiment directory.

    Returns:
        A list of Experiment objects.
    """
    return [retrieval_experiment(os.path.join(exp_path, f)) for f in os.listdir(exp_path)]


def list_test_names_from_experiment(experiment_name: str) -> List[str]:
    """
    Returns a list of test names found in the specified experiment directory.

    Args:
        experiment_name: The name of the experiment directory.

    Returns:
        A list of test names.
    """
    return os.listdir(os.path.join(exproot(), experiment_name))


def find_path_from_test_name(test_name: str) -> str:
    """
    Returns the path of the specified test name.

    Args:
        test_name: The name of the test.

    Returns:
        The path of the test, or None if not found.
    """
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


def retrieval_test_root(test_flag: str) -> str:
    """
    Returns the test root directory for the specified test name or test root.

    Args:
        test_flag: The test name or test root.
        like `230214.037.62t` or path like `path/to/230214.037.62t`
    Returns:
        The test root directory, or None if not found.
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


def retrieval_experiment(test_name=None, test_root: str = None) -> Experiment:
    """
    Loads an Experiment object from disk for the given test name or test root.

    Args:
        test_name (str, optional): The name of the test to load. If not provided,
            the test root directory must be provided instead. Defaults to None.
        test_root (str, optional): The root directory of the test to load. If not
            provided, the root directory is determined from the test name using
            the retrieval_test_root function. Defaults to None.

    Returns:
        Optional[Experiment]: The loaded Experiment object, or None if the test
        root cannot be determined or the Experiment cannot be loaded from disk.
    """
    if test_root is None:
        test_root = retrieval_test_root(test_name)
    if test_root is None:
        return None
    exp = Experiment.from_disk(test_root)
    return exp


def summary_experiment(test_name: str = None, test_root: str = None):
    """
    Prints a summary of the experiment specified by test_name or test_root.

    Args:
        test_name: The name of the test.
        test_root: The path to the test root directory.
    """
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
    indent_print(pformat(exp.roots))
    print('Execute:')
    indent_print(' '.join(exp.exec_argv))
    print('-----------------------------------')


def format_experiment(exp: Experiment) -> Dict[str, Any]:
    """
    Formats the Experiment object into a dictionary.

    Args:
        exp: An Experiment object.

    Returns:
        A dictionary of the Experiment properties, tags, paths, and execution arguments.
    """
    return {
        'Properties': exp.properties,
        'tags': exp.tags,
        'paths': exp.roots,
        'exec_argv': exp.exec_argv,
    }


def list_all_metrics(metric_root=None) -> Dict[str, List[str]]:
    """
    Returns a dictionary of all metrics found under metric_root directory.

    Args:
        metric_root: The root directory to search for metrics. Default is None, which uses the default metric root directory.

    Returns:
        A dictionary of all metrics, where the keys are the metric names and the values are lists of corresponding metric files.
    """
    if metric_root is None:
        metric_root = metricroot()

    res = {}
    for root, dirs, fs in os.walk(metric_root):
        if root == metric_root:
            continue
        fs = [os.path.join(root, i) for i in fs if not i.startswith('.')]
        res[os.path.basename(root)] = fs
    return res
