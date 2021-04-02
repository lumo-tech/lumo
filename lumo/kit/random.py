"""

"""
import os
import time

from lumo.utils import safe_io as io
from lumo.utils import random
from random import randint

class Random:
    """
    用于管理随机种子
    """

    def __init__(self, save_dir="./rnd"):
        self.save_dir = save_dir

    def _int_time(self):
        """用于获取一个理论上不会重复的随机种子"""

        return int(str(time.time()).split(".")[-1]) + randint(0, 10000)

    def _save_rnd_state(self, name):
        """保存种子"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        seed = random.hashseed(name)
        stt = random.fix_seed(seed)

        io.dump_state_dict(stt,self._build_state_name(name))

    def _have_rnd_state(self, name) -> bool:
        """判断是否存在某个种子"""
        if not os.path.exists(self.save_dir):
            return False
        return os.path.exists(self._build_state_name(name))

    def _get_rnd_state(self, name):
        """获取某个种子"""
        if not self._have_rnd_state(name):
            return None
        return io.load_state_dict(self._build_state_name(name))

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

    def mark(self, name):
        """
        用于数据集读取一类的，需要特定步骤每一次试验完全相同
        Args:
            name: 该次标记固定种子的名字，第一次调用该方法会在特定目录存放当前状态，
            第二次调用会在该位置读取当前随机种子状态

        Returns:

        """
        stt = self._get_rnd_state(name)
        if stt is not None:
            random.set_state(stt)
            return True
        else:
            self._save_rnd_state(name)
            return False

    def shuffle(self, seed=None):
        if seed is None:
            random.fix_seed(self._int_time())
        else:
            random.fix_seed(seed)

    def list(self):
        """列出当前保存的所有种子"""
        return [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir) if f.endswith('rnd')]
