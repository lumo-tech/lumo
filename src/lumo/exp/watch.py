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
        return f'C({self.name} {self.op} {self.value})'

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
            exp_root = os.path.join(exproot(), 'hb')

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
            return pd.DataFrame()
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
                    except:
                        continue

        for exp_name, tests in updates.items():
            dic = PDict(os.path.join(self.db_root, f'{exp_name}.sqlite'))

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

    def progress(self, is_alive=True):
        """
        Returns a DataFrame of alive experiments.

        Args:
            is_alive (bool): A boolean flag indicating whether to return only alive experiments.

        Returns:
            A pandas DataFrame containing the experiment information of alive experiments.
        """
        res = []
        for root, dirs, fs in os.walk(self.pid_root):
            for f in fs:
                if not f.endswith('.pid'):
                    continue
                try:
                    test_root = IO.load_text(os.path.join(root, f))
                    exp = Experiment.from_disk(test_root)
                    if exp.is_alive == is_alive:
                        res.append(exp.dict())
                    if not exp.is_alive and exp.properties['progress'].get('finished', None) is None:
                        exp.dump_info('progress',
                                      {
                                          'end': strftime(), 'finished': True, 'end_code': -1,
                                          'msg': 'ended by watcher'}
                                      )
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

    def widget(self,
               is_finished: bool = None,
               is_alive: bool = None,
               time_filter: list = None,
               params_filter: list = None,
               metric_filter: list = None
               ):
        """Create a user interface in jupyter with ipywidget"""
        assert params_filter is None or isinstance(params_filter, list)
        assert metric_filter is None or isinstance(metric_filter, list)

        from ipywidgets import widgets, interact, Label
        from IPython.display import display

        def make_row(dic: dict):
            """
            Helper function for creating a row in the widgets grid.

            Args:
                dic (dict): The dictionary containing the experiment information.

            Returns:
                A list of ipywidgets objects for the row.
            """
            exp = Experiment.from_cache(dic.copy())

            def on_note_update(sender):
                """when note textarea update"""
                exp.dump_note(sender['new'])

            def on_tag_update(sender):
                """when tag component update"""
                exp.dump_tags(*sender['new'])

            note_ui = widgets.Textarea(dic['note'])

            note_ui.continuous_update = False
            note_ui.observe(on_note_update, names='value', type='change')

            tags = dic.get('tags', [])
            try:
                tags = list(tags)
            except:
                tags = []
            tag_ui = widgets.TagsInput(value=tags)
            tag_ui.observe(on_tag_update, names='value', type='change')

            now = datetime.now()
            start = strptime(datestr=dic['progress']['start'])
            end = strptime(datestr=dic['progress']['last_edit_time'])

            human = widgets.VBox([

            ])
            return [
                widgets.Label(dic['exp_name']),
                widgets.Label(dic['test_name']),
                widgets.Label(f"""{strftime('%y-%m-%d %H:%M:%S', dateobj=start)}"""),
                widgets.Label(f"""{strftime('%y-%m-%d %H:%M:%S', dateobj=end)}"""),
                widgets.HTML('\n'.join([
                    f'{k}: {v}'
                    for k, v in dic['metrics'].items()
                    if isinstance(v, numbers.Number)
                ])),
                widgets.HBox([note_ui,
                              tag_ui, ])

            ]

        test_status = widgets.RadioButtons(options=['full', 'running', 'failed', 'succeed', 'finished'])
        start_filter = widgets.DatetimePicker()
        end_filter = widgets.DatetimePicker()

        def status_filter(sender):
            """when filter condition changed"""
            print(sender)
            make()

        test_status.observe(status_filter, names='value', type='change')

        # display()

        @interact
        def make(
                status=widgets.RadioButtons(options=['full', 'running', 'failed', 'succeed', 'finished']),
                start=widgets.DatetimePicker(),
                end=widgets.DatetimePicker(),
        ):
            """make widgets with filter condition."""
            if status == 'running':
                df = self.progress()
            elif status == 'finished':
                df = self.progress(is_alive=False)
            else:
                df = self.load()
                if status == 'succeed':
                    df = df[df['progress'].apply(lambda x: x['finished'])]
                elif status == 'failed':
                    df = df[df['exception'].isna() == False]

            if start:
                df = df.pipe(
                    lambda x: x[x['progress'].apply(lambda y: strptime(datestr=y['start'])) > start]
                )
            if end:
                df = df.pipe(
                    lambda x: x[x['progress'].apply(lambda y: strptime(datestr=y['end'])) < end]
                )

            if params_filter is not None:
                df_params = df['params']
                masks = None
                for condition in params_filter:
                    mask = condition.mask(df_params)
                    if masks is None:
                        masks = mask
                    else:
                        masks *= mask
                df = df[masks]

            if metric_filter is not None:
                df_params = df['metrics']
                masks = None
                for condition in metric_filter:
                    mask = condition.mask(df_params)
                    if masks is None:
                        masks = mask
                    else:
                        masks *= mask
                df = df[masks]

            exps = df.to_dict(orient='records')
            # grid = widgets.GridspecLayout(len(exps) + 1, 7)

            children = [
                widgets.Label('exp_name'),
                widgets.Label('test_name'),
                widgets.Label('start'),
                widgets.Label('end'),
                widgets.Label('metrics'),
                widgets.Label('note & tags'),
            ]
            # grid[0, 0] = widgets.Label('Meta')
            # grid[0, 1] = widgets.Label('Metrics')
            # grid[0, 2] = widgets.Label('Notes')
            for i, exp in enumerate(exps, start=1):
                row = make_row(exp)
                children.extend(row)
                # display(widgets.HBox(row))
                # for j, item in enumerate(row):
                #     grid[i, j] = item

            grid = widgets.GridBox(children=children,

                                   layout=widgets.Layout(
                                       width='100%',
                                       grid_template_columns=' '.join(['auto'] * 5) + ' auto',
                                       # grid_template_rows='80px auto 80px',
                                       grid_gap='5px 10px')
                                   )
            display(
                widgets.HTML("""
                <style>
                .widget-gridbox div:nth-of-type(even) {background-color: #f2f2f2 !important;}
                </style>
                """),
                grid,

            )


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
