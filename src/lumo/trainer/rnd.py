from typing import Union

from lumo.utils import random


class RndManager:
    """
    A seed manager for the trainer. Provides an interface for `~lumo.utils.random`.
    """

    def mark(self, seed: Union[int, str]):
        """
        Fixes the random seed to a specific state for reproducibility.

        Args:
            seed (Union[int, str]): The name of the fixed seed state.
        """
        random.fix_seed(random.hashseed(seed))

    def shuffle(self, seed=None):
        """
        Shuffles the random seed for reproducibility.

        Args:
            seed (int, optional): The random seed to use. If None, a random seed based on the current
            time will be used.
        """
        if seed is None:
            random.fix_seed(random.int_time())
        else:
            random.fix_seed(seed)
