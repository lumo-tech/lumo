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
import numbers
import os
import os.path
import re
from typing import List, Dict, overload
from pprint import pformat

from dbrecord import PDict
from datetime import datetime
from operator import gt, ge, le, lt, eq, ne

from lumo.proc.path import progressroot, exproot, dbroot, cache_dir
from .experiment import Experiment
from lumo.utils import safe_io as IO
from lumo.utils.fmt import format_timedelta, strptime, strftime
from lumo.proc.tz import timezone
from lumo.proc import glob

PID_ROOT = os.path.join(progressroot(), 'pid')
HB_ROOT = os.path.join(progressroot(), 'hb')
EXP_ROOT = os.path.join(progressroot())

styles = {
    'row-radio': """<style>
    .widget-radio-box {flex-direction: row !important;} 
    .widget-radio-box label{margin-bottom:10px !important;}
    </style>""",
    'widget-box': """
    <style>
    .widget-box {board: 1px solid;}
    </style>
    
    """
}


def in_(ser, value):
    """pandas operation"""
    return ser.apply(lambda x: x in value)


def not_in_(ser, value):
    """pandas operation"""
    return ser.apply(lambda x: x not in value)


# supported conditions
mapping = {
    '>=': ge,
    '<=': le,
    '==': eq,
    '!=': ne,
    '>': gt,
    '<': lt,
    'in': in_,
    'notin': not_in_,
}


class Condition:
    """Represents a condition to filter data based on a certain criteria."""

    def __init__(self, name: str = None, value=None, op=None):
        self.name = name
        self.value = value
        self.op = op

    def __getattr__(self, item):
        return Condition(item)

    def __getitem__(self, item):
        return Condition(item)

    def __neg__(self):
        self.drop = True
        return self

    def __ge__(self, other):
        if other is None:
            raise AssertionError()
        self.value = other
        self.op = ">="
        return self

    def __le__(self, other):
        if other is None:
            raise AssertionError()
        self.value = other
        self.op = "<="
        return self

    def __eq__(self, other):
        self.value = other
        self.op = "=="
        return self

    def __ne__(self, other):
        self.value = other
        self.op = "!="
        return self

    def __gt__(self, other):
        if other is None:
            raise AssertionError()
        self.value = other
        self.op = ">"
        return self

    def __lt__(self, other):
        assert other is not None
        self.value = other
        self.op = "<"
        return self

    def __repr__(self):
        return f'C({self.name}, {self.value}, {self.op})'

    def in_(self, lis):
        """
        Sets the condition to evaluate if the value is in a given list.

        Args:
            lis (list): the list of values to compare against.

        Returns:
            The current instance of the Condition class with the comparison operator and value set.
        """
        self.op = 'in'
        self.value = set(lis)
        return self

    def not_in_(self, lis):
        """
        Sets the condition to evaluate if the value is not in a given list.

        Args:
            lis (list): the list of values to compare against.

        Returns:
            The current instance of the Condition class with the comparison operator and value set.
        """
        self.op = 'notin'
        self.value = set(lis)
        return self

    def mask(self, df):
        """
        Returns a boolean mask of the given DataFrame based on the condition.

        Args:
            df (pd.DataFrame): the DataFrame to evaluate.

        Returns:
            A boolean mask of the given DataFrame based on the condition.
        """
        import pandas as pd
        names = self.name.split('.')
        value = df
        for i in names:
            if isinstance(value, pd.DataFrame):
                value = value[i]
            else:
                value = df.apply(lambda x: x[i])
        return mapping[self.op](value, self.value)

    def apply(self, df):
        """Returns a new DataFrame with only the rows that meet the condition."""
        return df[self.mask(df)]


C = Condition()


class Watcher:
    """
    A class for listing and watching experiments with time order and caching test information in 'metrics/<experiment>.sqlite'.

    Attributes:
        exp_root (str): The root directory to search for experiments.
        hb_root (str): The root directory to search for heartbeat files.
        pid_root (str): The root directory to search for PID files.
        db_root (str): The root directory to store the experiment databases.
    """

    def __init__(self, exp_root=None, hb_root=None, pid_root=None, db_root=None):
        if exp_root is None:
            exp_root = exproot()

        if hb_root is None:
            hb_root = os.path.join(cache_dir(), 'heartbeat')

        if pid_root is None:
            pid_root = os.path.join(cache_dir(), 'pid')

        if db_root is None:
            db_root = dbroot()
        self.db_root = db_root
        self.exp_root = exp_root
        self.hb_root = hb_root
        self.pid_root = pid_root

    def retrieve(self, test_name=None):
        raise NotImplementedError()

    def update(self):
        """Diff & Update"""
        updates = {}
        if not os.path.exists(self.hb_root):
            return {}
        for root, dirs, fs in os.walk(self.hb_root):
            if root == self.hb_root:
                continue
            for f in fs:
                if f.endswith('hb'):
                    hb_file = os.path.join(root, f)
                    test_root = IO.load_text(hb_file)
                    try:
                        exp = Experiment.from_disk(test_root)
                        updates.setdefault(exp.exp_name, []).append(exp.cache())
                    except KeyboardInterrupt as e:
                        raise e
                    except Exception as e:
                        print(e)
                        continue
                    finally:
                        os.remove(hb_file)

        for exp_name, tests in updates.items():
            dic = PDict(os.path.join(self.db_root, f'{exp_name}.sqlite'))

            for test in tests:
                dic[test['test_name']] = test
            dic.flush()
        return updates

    def fullupdate(self):
        updates = {}
        for root, dirs, fs in os.walk(self.exp_root):
            for f in dirs:
                if is_test_name(f):
                    info_dir = os.path.join(root, f)
                    try:
                        exp = Experiment.from_disk(info_dir)
                        updates.setdefault(exp.exp_name, []).append(exp.cache())
                    except KeyboardInterrupt as e:
                        raise e
                    except Exception as e:
                        print(e)
                        continue

        for exp_name, tests in updates.items():
            dic = PDict(os.path.join(self.db_root, f'{exp_name}.sqlite'))
            dic.clear()

            for test in tests:
                dic[test['test_name']] = test
            dic.flush()

        return updates

    def load(self, with_pandas=True):
        """
        Loads the experiment information from heartbeat files and the experiment databases.

        Args:
            with_pandas (bool, optional): whether to return the experiment information as a pandas DataFrame.
                Defaults to True.

        Returns:
            If with_pandas is True, returns a pandas DataFrame containing the experiment information
            sorted by experiment name and test name. Otherwise, returns a dictionary containing the
            experiment information.
        """
        res = {}
        updates = self.update()

        def valid_row(dic):
            return isinstance(dic, dict) and 'test_name' in dic

        for dic_fn in os.listdir(self.db_root):
            if not dic_fn.endswith('sqlite'):
                continue
            dic = PDict(os.path.join(self.db_root, dic_fn))
            exp_name = os.path.splitext(dic_fn)[0]

            for test_name, test_prop in dic.items():
                if valid_row(test_prop):
                    res[test_name] = test_prop

            if exp_name in updates:
                for test in updates[exp_name]:
                    if valid_row(test):
                        res[test['test_name']] = test
                dic.flush()

        if with_pandas:
            try:
                import pandas as pd
            except ImportError as e:
                print(
                    'with_padnas=True requires pandas to be installed, use pip install pandas or call `.load(with_padnas=False)`')

            df = pd.DataFrame(res.values())
            df = df.sort_values(['exp_name', 'test_name'])
            return df.reset_index(drop=True)
        else:
            return res

    def progress(self, with_pandas=True):
        """
        Returns a DataFrame of alive experiments.

        Returns:
            A pandas DataFrame containing the experiment information of alive experiments.
        """
        res = []
        for root, dirs, fs in os.walk(self.pid_root):
            for f in fs:
                if not f.endswith('.pid'):
                    continue
                try:
                    pid_f = os.path.join(root, f)
                    test_root = IO.load_text(pid_f)
                    exp = Experiment.from_disk(test_root)

                    if exp.is_alive:
                        res.append(exp.dict())
                    elif exp.properties['progress'].get('end_code', None) is None:
                        if (datetime.timestamp(datetime.now()) - os.stat(pid_f).st_mtime) < glob.get(
                                'ALIVE_SECONDS', 60 * 5):  # powered by exp/agent.py
                            res.append(exp.dict())
                    else:
                        exp.dump_info('progress',
                                      {
                                          'end': strftime(), 'end_code': -10,
                                          'msg': 'ended by watcher'}
                                      )
                        os.remove(pid_f)
                except:
                    continue
        if with_pandas:
            import pandas as pd
            return pd.DataFrame(res)
        else:
            return res

    def interactive(self):
        """interactive, mark, label, note in ipython environment."""
        pass

    def server(self):
        """simple server which make you note your experiments"""
        pass

    def panel(self):
        from .lazy_panel import make_experiment_tabular
        widget = make_experiment_tabular(self.load(), self.load)
        return widget


def is_test_root(path: str) -> bool:
    """
    Determines if the specified path is a valid test root.

    Args:
        path: The path to check.

    Returns:
        True if the path is a valid test root, False otherwise.
    """
    test_name = os.path.basename(path.rstrip('/'))
    return is_test_name(test_name)


def is_test_name(test_name: str) -> bool:
    """
    Determines if the specified string is a valid test name.

    Args:
        test_name: The string to check.

    Returns:
        True if the string is a valid test name, False otherwise.
    """
    return re.search(r'^\d{6}\.\d{3}\.[a-z\d]{2}t$', test_name) is not None
