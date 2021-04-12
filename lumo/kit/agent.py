import time

import psutil

from lumo.kit.experiment import Experiment
from lumo.kit.params import BaseParams
from lumo.utils.dates import strftime
from lumo.utils.keys import EXP


def wait_pid_stop():
    params = BaseParams()
    params.pid = None
    params.test_name = None
    params.exp_name = None
    params.from_args()

    exp = Experiment(params.exp_name, params.test_name)
    c = 0
    while psutil.pid_exists(params.pid):
        info = exp.load_info(EXP.STATE)
        if 'end_code' in info:
            break

        exp.dump_info(EXP.STATE, {
            'end': strftime(),
        }, append=True)
        time.sleep(1)

if __name__ == '__main__':
    wait_pid_stop()
