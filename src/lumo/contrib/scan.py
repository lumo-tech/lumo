"""
基线对比
"""
import time
from itertools import cycle
from typing import List
import torch

from lumo import Params, Logger
import sys

DATE_FLAG = '2023.03.04'


class ScanBaseParams(Params):

    def __init__(self):
        super().__init__()
        self.gpus = None
        self.group = None
        self.interval = 5  # sleep interval between two tests
        self.skip = 0
        self.n = -1


def format_args(**kwargs):
    return ' '.join([f'--{k}={v}' for k, v in kwargs.items()])


def base_main(pm: ScanBaseParams, files: List[str], dics: List[dict]):
    assert isinstance(pm, ScanBaseParams)
    log = Logger()
    log.use_stdout = False
    log.add_log_dir(f'./log_{pm.group}')

    base = ("sleep {sleep} ; " +
            sys.executable +
            " {file} {kwargs} --device={device}{group} & \n"
            )

    if not torch.cuda.is_available():
        gpus = ['cpu']
    elif pm.gpus is None:
        gpus = list(range(torch.cuda.device_count()))
    elif isinstance(pm.gpus, (int, str)):
        gpus = [torch.device(pm.gpus).index]
    else:
        gpus = pm.gpus

    # append None to identity loop end.
    gpus.append(None)
    gpus = cycle(gpus)
    c = 0
    cnt = 0
    for i, (file, kwargs) in enumerate(zip(files, dics)):
        if pm.skip > cnt:
            cnt += 1
            continue

        device = next(gpus)
        if device is None:
            # wait until all devices are free.
            c = 0
            print('wait', flush=True)
            device = next(gpus)

        if pm.group is not None:
            group = f" --group={pm.group} "
        else:
            group = ''

        cur = base.format(
            sleep=c * pm.interval,
            file=file, kwargs=format_args(**kwargs),
            device=device, group=group)

        c += 1
        cnt += 1
        log.info(cur.strip())
        print(cur, flush=True, end='')

        if pm.n > 0 and pm.skip + pm.n <= cnt:
            break

    print('wait', flush=True)
