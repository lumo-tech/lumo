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


def wait_pid_stop(info_dir=None):
    """wait test """
    params = BaseParams()
    params.info_dir = info_dir
    params.from_args()

    exp = Experiment.from_disk(params.info_dir)
    pid = exp.properties['pinfo'].get('pid')
    info_dir = exp.info_dir
    try:
        while pid is not None and psutil.pid_exists(pid):
            exp.trigger()
            exp.dump_info('agent', {'last_edit_time': strftime()})

            time.sleep(10)
    except:
        pass


if __name__ == '__main__':
    wait_pid_stop()
