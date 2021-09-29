"""

"""
import time

import psutil

from lumo.kit.experiment import Experiment
from lumo.proc.const import EXP_CONST
from lumo.kit.params import BaseParams
from lumo.proc.date import strftime


def wait_pid_stop():
    params = BaseParams()
    params.pid = None
    params.test_name = None
    params.exp_name = None
    params.state_key = None
    params.from_args()

    exp = Experiment(params.exp_name, params.test_name)
    while psutil.pid_exists(params.pid):
        info = exp.load_info(EXP_CONST.INFO_KEY.STATE)
        if 'end_code' in info:
            break

        exp.dump_info(params.state_key, {
            'end': strftime(),
        }, append=True)
        time.sleep(1)


if __name__ == '__main__':
    wait_pid_stop()
