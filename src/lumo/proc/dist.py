import os

from torch import distributed as dist

__all__ = ['local_rank', 'world_size', 'is_dist', 'is_main']


def local_rank(group=None):
    """
    Returns the rank of the current process in the provided ``group`` or the
    default group if none was provided.

    Args:
        group (ProcessGroup, optional): See `torch.distributed.get_rank()` for details.

    Returns:
        The rank of the process group
        -1, if not part of the group

    See Also:
        `torch.distributed.get_rank()`
    """
    rank = int(os.environ.get('LOCAL_RANK', '-1'))
    if rank == -1:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank(group)

    return rank


def world_size(group=None):
    """
    Returns the number of processes in the current process group

    Args:
        group (ProcessGroup, optional): See `torch.distributed.get_world_size()` for details.

    Returns:
        The world size of the process group
        -1, if not part of the group

    See Also:
        `torch.distributed.get_rank()`
    """

    size = int(os.environ.get('WORLD_SIZE', '0'))
    if size == 0:
        if dist.is_available() and dist.is_initialized():
            size = dist.get_world_size(group)
    return size


def is_dist() -> bool:
    """
    Whether the experiment is running in a distributed manner.

    Returns:
        True if it is a distributed program.
        False for not.

    """
    return local_rank() >= 0


def is_main():
    """
    Whether the experiment is the main process.

    Returns:
        True if:
        - Current process is running in distributed manner and local_rank()==0, or
        - not a distributed process.

        False for others.

    """
    return local_rank() <= 0
