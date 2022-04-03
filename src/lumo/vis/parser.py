import os
from itertools import chain
from typing import Dict, List

try:
    import regex as re
except ImportError:
    import re

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from lumo.exp import Experiment, finder

match_metric_v0 = re.compile(' ([a-z0-9A-Z_]+): ([0-9.]+)')
match_metric_v1 = re.compile(' ([a-z0-9A-Z_])=([0-9.]+)')


@dataclass()
class Step:
    value: Number
    step: int


def find_metric_fron_test_root(test_root):
    test_root = finder.ensure_test_root(test_root)
    if test_root is None:
        return False, {}

    exp = Experiment.from_disk(test_root)
    if exp.has_prop('tensorboard_args'):
        tb = exp.get_prop('tensorboard_args')
        metrics = parse_fron_tensorboard(tb['log_dir'])
    elif exp.has_prop('logger_args'):
        tb = exp.get_prop('logger_args')
        metrics = parse_from_log(tb['log_dir'])
    else:
        fs = [i for i in os.listdir(exp.test_root)]
        if len([f for f in fs if f.endswith('.log')]) > 0:
            metrics = parse_from_log(os.path.join(exp.test_root, fs[0]))
        else:
            metrics = {}
    return True, metrics


def parse_from_log(log) -> Dict[str, List[Step]]:
    metrics = defaultdict(list)
    with open(log) as r:
        for line in r:
            temp1 = match_metric_v0.findall(line.strip())
            temp2 = match_metric_v0.findall(line.strip())
            for k, v in chain(temp1, temp2):
                try:
                    metrics[k].append(Step(value=float(v), step=len(metrics[k])))
                except ValueError:
                    pass
    return metrics


def parse_fron_tensorboard(root) -> Dict[str, List[Step]]:
    from lumo.vis.parser_tb import TBReader
    reader = TBReader(root)
    metrics = defaultdict(list)
    for k in reader.scalar_names:
        metrics[k] = [Step(i.value, i.step) for i in reader.Scalars(k)]
    return metrics
