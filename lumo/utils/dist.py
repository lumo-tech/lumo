import os
from torch import distributed as dist


def local_rank():
    rank = os.environ.get('LOCAL_RANK', -1)
    if rank == -1:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

    return int(rank)


def world_size():
    size = os.environ.get('WORLD_SIZE', 0)
    if size == 0:
        if dist.is_available() and dist.is_initialized():
            size = dist.get_world_size()
    return int(size)


def is_dist():
    return local_rank() >= 0
