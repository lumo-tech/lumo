import numbers
import os
import os.path
import re
from typing import List

from dbrecord import PDict
from datetime import datetime
from operator import gt, ge, le, lt

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


def eq(ser, value):
    if value is None:
        return ser.isna() == True
    else:
        return ser == value


def ne(ser, value):
    if value is None:
        return ser.isna() == False
    else:
        return ser != value


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
    """
    Represents a condition to filter data based on a certain criteria.

    row filter:
    ```
    from lumo import C
    import pandas as pd

    # create a sample DataFrame
    data = {'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
            'age': [25, 30, 35, 40, 45],
            'city': [{'index':0}, {'index':1}, {'index':2},{'index':3},{'index':4}]}
    df = pd.DataFrame(data)

    # create and apply the condition to filter the DataFrame
    filtered_df = (C['age'] >= 35).apply(df)

    # print the filtered DataFrame
    print(filtered_df)
    ```

    column edit:
    ```
    (C+{'city.index':'cindex'}).apply(df).columns
    # Index(['name', 'age', 'city', 'cindex'], dtype='object')

    (-C['city']).apply(df).columns
    # Index(['name', 'age'], dtype='object')

    (C-['city']).apply(df).columns
    (C-'city').apply(df).columns
    # Index(['name', 'age'], dtype='object')

    (C-['city','name']).apply(df).columns
    # Index(['age'], dtype='object')
    ```

    pipeline:
    ```
    C.pipe(df,[
        (C['age']>35),
        C+{'city.index':'cindex'},
        C-['city','name']
    ])
    ```
    """

    def __init__(self, name: str = None, value=None, op=None):
        self.name = name
        self.value = value
        self.op = op

    def __getattr__(self, item):
        return Condition(item)

    def __getitem__(self, item):
        return Condition(item)

    def __add__(self, other):
        c = Condition()
        c.op = 'add_column'
        c.value = {}
        if isinstance(other, str):
            c.value[other] = other
        elif isinstance(other, dict):
            c.value.update(other)
        else:
            raise NotImplementedError()
        return c

    def __sub__(self, other):
        c = Condition()
        c.name = None
        c.op = 'drop_column'
        c.value = {}
        if isinstance(other, str):
            c.value.update({other: None})
        elif isinstance(other, (list, set, dict)):
            c.value.update({k: None for k in other})
        else:
            raise NotImplementedError()
        return c

    def __neg__(self):
        c = Condition()
        c.op = 'drop_column'
        return c

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
                value = value.apply(lambda x: x.get(i) if isinstance(x, dict) else None)
        return mapping[self.op](value, self.value)

    def capply(self, df):
        """apply operations in column axis"""
        import pandas as pd
        df = df.reset_index(drop=True)
        if self.op == 'drop_column':
            if isinstance(self.name, str):
                var = [self.name]
            elif isinstance(self.value, str):
                var = [self.value]
            else:
                var = list(self.value)
            print(var)
            print(df.columns)
            df = df.drop(var, axis=1)
        else:
            assert isinstance(self.value, dict)
            for name, aim in self.value.items():
                names = name.split('.')
                value = df
                for i in names:
                    if isinstance(value, pd.DataFrame):
                        value = value[i]
                    else:
                        value = value.apply(lambda x: x.get(i) if isinstance(x, dict) else None)
                df[aim] = value
        return df

    def apply(self, df):
        """Returns a new DataFrame with only the rows that meet the condition."""
        if not self.op.endswith('column'):
            return df[self.mask(df)].reset_index(drop=True)
        else:
            return self.capply(df)

    def pipe(self, df, conditions: List['Condition']):
        """Applies a list of conditions to a DataFrame using the pipe method."""
        filtered_df = df
        for condition in conditions:
            filtered_df = condition.apply(filtered_df)  # .reset_index(drop=True)
        return filtered_df


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

    def retrieve(self, test_name=None) -> Experiment:
        """retrieve the Experiment of given test name"""
        df = self.load()
        res = df[df['test_name'] == test_name]
        if len(res) > 0:
            return Experiment.from_cache(res.iloc[0].to_dict())
        return None

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
        """re-update all experiment from original experiment directories"""
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
            """is a valid row?"""
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
                                      , append=True
                                      )
                        os.remove(pid_f)
                except:
                    continue
        if with_pandas:
            import pandas as pd
            return pd.DataFrame(res)
        else:
            return res

    def server(self):
        """simple server which make you note your experiments"""
        pass

    def panel(self, df=None):
        """create a dashboard powered by Panel, need to install panel library first."""
        from .lazy_panel import make_experiment_tabular
        if df is None:
            df = self.load()
        widget = make_experiment_tabular(df)
        return widget


def is_test_root(path: str) -> bool:
    """
    Determines if the specified path is a valid test root.

    Args:
        path: The path to check.

    Returns:
        True if the path is a valid test root, False otherwise.
    """
    if os.path.exists(os.path.join(path, 'info', 'test_name.json')):
        return True

    # test_name = os.path.basename(path.rstrip('/'))
    # return is_test_name(test_name)


def is_test_name(test_name: str) -> bool:
    """
    Determines if the specified string is a valid test name.

    Args:
        test_name: The string to check.

    Returns:
        True if the string is a valid test name, False otherwise.
    """
    return re.search(r'^\d{6}\.\d{3}\.[a-z\d]{2}t$', test_name) is not None
