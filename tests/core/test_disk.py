from lumo.core.disk import Metrics, TableRow
import numpy as np
import tempfile
from lumo.proc.config import glob
from lumo.utils import safe_io as IO

glob['metric_root'] = tempfile.mkdtemp()
from lumo.utils.fmt import strftime


def test_table_row():
    row = TableRow('lumo-test', 'core', strftime())

    # test update
    row.update('a', 'b')
    assert row['a'] == 'b'

    # test update_metric
    ## test max
    assert row.update_metric('acc', 100, compare='max')['acc'] == 100
    assert row.update_metric('acc', 90, compare='max')['acc'] == 100
    assert row.update_metric('acc', 110, compare='max')['acc'] == 110
    assert row['metric']['acc'] == 110
    ## test min
    assert row.update_metric('loss', 0.2, compare='min')['loss'] == 0.2
    assert row.update_metric('loss', 0.3, compare='min')['loss'] == 0.2
    assert row.update_metric('loss', 0.1, compare='min')['loss'] == 0.1
    assert row['metric']['loss'] == 0.1

    # test update_metric_pair
    res = row.update_metric_pair('accE', 50, 'clsAcc', [0, 1, 2, 3], compare='max')
    assert res['accE'] == 50
    assert res['clsAcc'] == [0, 1, 2, 3]
    res = row.update_metric_pair('accE', 60, 'clsAcc', np.array([2, 3, 4, 5]), compare='max')
    assert res['accE'] == 60
    assert (res['clsAcc'] == np.array([2, 3, 4, 5])).all()

    # test storage
    row.flush()
    storage = IO.load_pkl(row.fpath)
    assert storage['a'] == row['a']
    assert storage['metric']['acc'] == row['metric']['acc']
    assert (storage['metric']['clsAcc'] == row['metric']['clsAcc']).all()
