import os.path
import shutil

from dbrecord import PDict
from lumo import Logger
from lumo.utils import safe_io as IO
from lumo.proc.path import metricroot
import pandas as pd


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


def collect_table_rows(metric_root=None):
    res = []
    logger = Logger()
    exp_map = list_all_metrics(metric_root)
    for k, rows in exp_map.items():
        # append existing row metrics
        global_dic = PDict(os.path.join(metricroot(), f'{k}.dict.sqlite'))
        for row in global_dic.values():
            res.append(row)

        if len(rows) == 0:
            continue

        logger.info(f'collecting {len(rows)} tests.')

        for row_fn in rows:
            if not row_fn.endswith('pkl'):
                continue
            try:
                row = IO.load_pkl(row_fn)
            except:
                print(f'Failed on load {row_fn}')
                continue
            test_name = os.path.splitext(os.path.basename(row_fn))[0]
            global_dic[test_name] = row
            shutil.move(row_fn, os.path.join(os.path.dirname(row_fn), f'.{test_name}.pkl'))
            res.append(row)
        global_dic.flush()

    return pd.DataFrame(res)


def replac(df: pd.DataFrame):
    """replace after filtering"""
    raise NotImplementedError()
