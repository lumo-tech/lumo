import os

from lumo import Experiment, Params
import tarfile
from lumo.utils import safe_io as IO
from lumo.proc.config import debug_mode
import tempfile


def get_experiment():
    debug_mode()
    exp = Experiment('test_exp')
    exp.start()
    exp.dump_info('temp_string', 'string')
    exp.dump_info('temp_number', 123)
    exp.dump_info('temp_list', [1, 2, 3])
    exp.dump_info('temp_dict', {"a": [1, 2, 3]})

    pfn = exp.mk_ipath('params.json')
    Params.init_from_kwargs(a=1, b=2, c=3).to_json(pfn)
    IO.dump_pkl({'a': 1}, exp.mk_bpath('bin.pkl'))
    IO.dump_pkl({'a': 1}, exp.mk_cpath('bin.pkl'))

    exp.end()

    return exp


def test_backup():
    exp = get_experiment()
    target_dir = tempfile.mkdtemp()

    fpath = exp.backup('local', target_dir=target_dir, with_blob=True, with_cache=True)
    file = tarfile.open(fpath, mode='r')
    de_dir = tempfile.mkdtemp()
    os.makedirs(de_dir, exist_ok=True)
    file.extractall(de_dir)
    bexp = Experiment.from_disk(
        os.path.join(de_dir, exp.test_name, 'info'),
        blob_dir=os.path.join(de_dir, exp.test_name, 'blob'),
        cache_dir=os.path.join(de_dir, exp.test_name, 'cache'),
    )

    assert bexp.properties['temp_string'] == 'string'
    assert bexp.properties['temp_number'] == 123
    assert bexp.properties['temp_list'] == [1, 2, 3]
    assert bexp.properties['temp_dict'] == {"a": [1, 2, 3]}

    assert IO.load_pkl(bexp.mk_bpath('bin.pkl'))['a'] == 1
    assert IO.load_pkl(bexp.mk_cpath('bin.pkl'))['a'] == 1
