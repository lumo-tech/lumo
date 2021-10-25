"""
heartbeat mechanism
"""
import time

import psutil

from lumo.kit.experiment import Experiment
from lumo.kit.params import BaseParams
from lumo.utils.fmt import strftime
from lumo.utils.safe_io import IO
from joblib import hash


def wait_pid_stop(exp_name=None, test_name=None, state_key='state1'):
    params = BaseParams()
    params.pid = None
    params.test_name = test_name
    params.exp_name = exp_name
    params.state_key = state_key
    params.from_args()

    exp = Experiment(params.exp_name, params.test_name)

    test_root = exp.test_root
    while params.pid is None or psutil.pid_exists(params.pid):
        exp.dump_info(params.state_key, {
            'end': strftime(),
        }, append=True)

        IO.dump_text(test_root, exp.root_branch.file(f'{hash(test_root)}.hb', 'heartbeat'))
        IO.dump_text(test_root, exp.exp_branch.file(f'{hash(test_root)}.hb', 'heartbeat'))
        IO.dump_text(test_root, exp.test_branch.file(f'{hash(test_root)}.hb', 'heartbeat'))

        time.sleep(2)


if __name__ == '__main__':
    wait_pid_stop('1.trainer.plusone')
