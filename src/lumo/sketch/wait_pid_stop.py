"""
heartbeat mechanism
"""
import time

import psutil
from joblib import hash

from lumo.core import BaseParams
from lumo.utils import safe_io as IO
from lumo.utils.fmt import strftime
from lumo.exp import Experiment


def wait_pid_stop(exp_name=None, test_root=None, state_key='state1'):
    params = BaseParams()
    params.pid = None
    params.exp_name = exp_name
    params.state_key = state_key
    params.from_args()

    exp = Experiment.from_disk(test_root)

    test_root = exp.test_root
    while params.pid is None or psutil.pid_exists(params.pid):
        exp.dump_info(params.state_key, {
            'end': strftime(),
        }, append=True)

        IO.dump_text(test_root, exp.root_branch.file(f'{hash(test_root)}.hb', 'heartbeat'))

        time.sleep(2)


if __name__ == '__main__':
    wait_pid_stop()
