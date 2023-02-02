import os
import pickle
import time
from joblib import hash
from lumo.proc.path import cache_dir
from lumo.utils import random


class RndManager:
    """
    用于管理随机种子
    """

    def __init__(self):
        self.save_dir = os.path.join(cache_dir(), 'rnd')

    def int_obj(self, obj):
        if isinstance(obj, (int, float, bool)):
            return int(obj)
        return int(hash(obj)[:4], 16)

    def mark(self, name):
        """
        用于数据集读取一类的，需要特定步骤每一次试验完全相同
        Args:
            name: 该次标记固定种子的名字，第一次调用该方法会在特定目录存放当前状态，
            第二次调用会在该位置读取当前随机种子状态

        Returns:

        """
        random.fix_seed(self.int_obj(name))

    def int_time(self):
        return int(str(time.time()).split(".")[-1])

    def shuffle(self, seed=None):
        """
        打乱，一般用于复现试验的时候随机一个种子
        Args:
            name:
            seed:

        Returns:

        """
        if seed is None:
            random.fix_seed(self.int_time())
        else:
            random.fix_seed(seed)

    def list(self):
        """列出当前保存的所有种子"""
        return [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir) if f.endswith('rnd')]

    def _save_rnd_state(self, name):
        """保存种子"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        seed = random.hashseed(name)
        stt = random.fix_seed(seed)

        with open(self._build_state_name(name), "wb") as f:
            pickle.dump(stt, f)

    def _have_rnd_state(self, name) -> bool:
        """判断是否存在某个种子"""
        if not os.path.exists(self.save_dir):
            return False
        return os.path.exists(self._build_state_name(name))

    def _get_rnd_state(self, name):
        """获取某个种子"""
        if not self._have_rnd_state(name):
            return None
        with open(self._build_state_name(name), "rb") as f:
            return pickle.load(f)

    def _build_state_name(self, name, replacement=False):
        if replacement:
            i = 1
            fn = os.path.join(self.save_dir, "{}.{:02d}.rnd".format(name, i))
            while os.path.exists(fn):
                i += 1
                fn = os.path.join(self.save_dir, "{}.{:02d}.rnd".format(name, i))
        else:
            fn = os.path.join(self.save_dir, "{}.rnd".format(name))

        return fn
