import os
import time
from typing import Union

from joblib import hash
from lumo.proc.path import cache_dir
from lumo.utils import random


class RndManager:
    """
    A seed manager for trainer. Provide interface for `~lumo.utils.random`
    """

    def __init__(self):
        self.save_dir = os.path.join(cache_dir(), 'rnd')

    def mark(self, seed: Union[int, str]):
        """
        用于数据集读取一类的，需要特定步骤每一次试验完全相同
        Args:
            seed: 该次标记固定种子的名字，第一次调用该方法会在特定目录存放当前状态，
            第二次调用会在该位置读取当前随机种子状态

        Returns:

        """
        random.fix_seed(random.hashseed(seed))

    def shuffle(self, seed=None):
        """
        打乱，一般用于复现试验的时候随机一个种子
        Args:
            name:
            seed:

        Returns:

        """
        if seed is None:
            random.fix_seed(random.int_time())
        else:
            random.fix_seed(seed)
