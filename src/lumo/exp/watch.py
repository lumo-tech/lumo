"""
Watcher 可以在运行实验后在 jupyter 或者网页上展示正在运行和已经运行结束的实验（按时间顺序？）
以及可以简化记录实验的烦恼

现在的核心痛点是
 - [ ] 所有元信息都有了，但是找不到哪个实验是哪个实验
 - [ ] 同时跑的多个实验有一个失败了，重跑时会混淆，或许需要一种覆盖手段 ->
 - > 怎么 rerun？
        lumo rerun test_name √
        lumo note html （用 streamlit 之类的生成动态网页）
        lumo note cmd  (类似 top 的视角，按时间顺序排列)
- > rerun 将已经跑的实验 move

可以代替 analysis 的作用。主要有

-> 按照 progress 目录，获取所有的实验
-> 根据获取的实验，按顺序记录
-> 每次只记录

"""
import os.path
from typing import List, Dict

import pandas as pd
from dbrecord import PDict

from lumo.proc.path import progressroot, exproot, dbroot
from .experiment import Experiment
from .finder import is_test_name
from lumo.utils import safe_io as IO
from lumo.analyse.collect import collect_table_rows

PID_ROOT = os.path.join(progressroot(), 'pid')
HB_ROOT = os.path.join(progressroot(), 'hb')
EXP_ROOT = os.path.join(progressroot())


class Watcher:
    """List and watch experiments with time order

    Cache test_information in
    metrics/<experiment>.sqlite
    """

    def __init__(self, exp_root=None, hb_root=None, pid_root=None, db_root=None):
        if exp_root is None:
            exp_root = os.path.join(exproot(), 'hb')

        if hb_root is None:
            hb_root = os.path.join(progressroot(), 'hb')

        if pid_root is None:
            pid_root = os.path.join(progressroot(), 'pid')

        if db_root is None:
            db_root = dbroot()
        self.db_root = db_root
        self.exp_root = exp_root
        self.hb_root = hb_root
        self.pid_root = pid_root

    def load(self):
        res = {}
        updates = {}
        for root, dirs, fs in os.walk(self.hb_root):
            if root == self.hb_root:
                continue
            for f in fs:
                if f.endswith('heartbeat'):
                    hb_file = os.path.join(root, f)
                    test_root = IO.load_text(hb_file)
                    try:
                        exp = Experiment.from_disk(test_root)
                        updates.setdefault(exp.exp_name, []).append(exp.dict())
                    except KeyboardInterrupt as e:
                        raise e
                    except:
                        continue

        for exp_name, tests in updates.items():
            dic = PDict(os.path.join(self.db_root, f'{exp_name}.sqlite'))
            for test_name, test_prop in dic.items():
                res[test_name] = test_prop

            for test in tests:
                dic[test['test_name']] = test
                res[test['test_name']] = test
            dic.flush()

        df = pd.DataFrame(res.values())
        return df

    def progress(self):
        """return the alive process"""
        res = []
        for pid in os.listdir(self.pid_root):
            try:
                test_root = IO.load_text(os.path.join(self.pid_root, pid))
                exp = Experiment.from_disk(test_root)
                res.append(exp.dict())
            except:
                continue
        return pd.DataFrame(res)

    def interactive(self):
        """interactive, mark, label, note in ipython environment."""
        pass

    def server(self):
        """simple server which make you note your experiments"""
        pass

    def list_all(self, exp_root=None, limit=100) -> Dict[str, List[Experiment]]:
        """
        Returns a dictionary of all experiments under exp_root directory.

        Args:
            exp_root: The root directory to search for experiments. Default is None, which uses the default experiment root directory.

        Returns:
            A dictionary of all experiments, where the keys are the names of the experiments and the values are lists of corresponding Experiment objects.
        """
