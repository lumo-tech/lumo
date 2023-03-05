"""
Methods about random
"""
import hashlib
import random
from typing import Union

import numpy as np
import torch
import time


def int_time():
    """
    Get the current time as an integer.

    Returns:
        int: The current time as an integer.
    """
    return int(str(time.time()).split(".")[-1])


def hashseed(hashitem: Union[int, str]):
    """
    Generate a hash seed from a given integer or string.

    Args:
        hashitem (Union[int, str]): The integer or string to generate the hash seed from.

    Returns:
        int: The hash seed.

    Raises:
        AssertionError: If the given `hashitem` is not an integer or a string.
    """
    if not isinstance(hashitem, (int, str)):
        raise AssertionError()

    if isinstance(hashitem, str):
        digest = hashlib.md5(hashitem.encode(encoding='utf-8')).digest()
        return sum([int(i) for i in digest])

    return hashitem


def fix_seed(seed=10, cuda=True):
    """

    Args:
        seed:

    Returns:

    Notes:
        When use dataloader and its num_workers is bigger than one, the final results may can't be the same cased by multithread.

        [2023.02.22] Currently (as MPS support is quite new) there is no way to set the seed for MPS directly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available() and cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # use determinisitic algorithm

    return get_state()


def fix_cuda():
    """
    Set deterministic and reproducible configuration for CUDA.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True


def get_state(cuda=True):
    """
    Get the current state of the random number generators.

    Args:
        cuda (bool): Whether to get the CUDA state if PyTorch is using CUDA.

    Returns:
        dict: A dictionary containing the current states of the random number generators for
        numpy, pytorch, pytorch.cuda, and python's built-in `random` module.

    """
    return {
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch.cuda": torch.cuda.get_rng_state() if (torch.cuda.is_available() and cuda) else None,
        "random": random.getstate(),
    }


def set_state(state_dict, cuda=True):
    """
    Set the random state for NumPy, PyTorch, PyTorch CUDA, and Python's built-in `random` module.

    Args:
        state_dict (dict): A dictionary containing the desired states of the random number generators for NumPy, PyTorch, PyTorch CUDA, and Python's built-in `random` module.
        cuda (bool): Whether to set the CUDA state if PyTorch is using CUDA.

    """
    random.setstate(state_dict["random"])
    np.random.set_state(state_dict["numpy"])
    torch.random.set_rng_state(state_dict["torch"])
    if torch.cuda.is_available() and cuda:
        if "torch.cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["torch.cuda"])
        else:
            import warnings
            warnings.warn("Don't have torch.cuda random state")

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
