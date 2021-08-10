import os
from torch import distributed as dist


def local_rank() -> int:
    """
    A safe function that always returns a number to identify the rank of current process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    rank = os.environ.get('LOCAL_RANK', -1)
    if rank == -1:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

    return int(rank)


def world_size() -> int:
    """
    A safe function that always returns the number of processes in the current process group

    Returns:
        The world size of the process group
        -1, if not part of the group
    Returns:

    """
    size = os.environ.get('WORLD_SIZE', 0)
    if size == 0:
        if dist.is_available() and dist.is_initialized():
            size = dist.get_world_size()
    return int(size)


def is_dist() -> bool:
    """
    Returns `True` if current program is in distribute mode, or `False` if not.
    """
    return local_rank() >= 0
