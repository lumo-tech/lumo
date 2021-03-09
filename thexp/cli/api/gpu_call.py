import torch
from typing import List, Union
from thexp.utils import re
import subprocess


def is_available():
    return torch.cuda.is_available()


def device_count():
    if not is_available():
        return 0

    return torch.cuda.device_count()


def all_memory_cached(device_id: Union[List[int], int] = None, process=False):
    from thexp.base_classes import llist
    match_mem = re.compile('([0-9]+[a-zA-Z]+) \/ ([0-9]+[a-zA-Z]+)')
    proc = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)

    count = device_count()
    res = llist()
    for i, line in enumerate(proc.stdout.readlines()):
        if i < 7:
            continue

        if i % 3 == 2:
            line = line.decode('utf-8').strip()
            res_ = re.findall(match_mem, line)
            res.append(res_[1][0])
            count -= 1
        if count == 0:
            break


    if device_id is not None:
        res = res[device_id]


    if process:
        # TODO 待验证
        match_num = re.compile('([0-9]+)')
        for i in range(len(res)):
            res[i] = re.findall(match_num,res[i])

    return res
