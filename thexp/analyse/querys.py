"""

Q = Query()

Q.repos()['SemiEnhance'].exps()['some_exp'].tests()[0,1,2]...
Q.repos()['SemiEnhance'].exps()['some_exp'].tests('0001','0002')...
Q.exps()[:2].tests()
Q.tests().first() -> TestViewer
Q.tests().last() -> TestViewer
Q.tests().last() -> TestViewer

    ... list all repos

blur search:
Q.rfind/efind/tfind/
Q.rfind('th')
Q.tfind('')

compare params:
Q.tests('xxx').compare_params()


compare files:
Q.tests('xxx').compare_files(include_current=True)

    output a table

Q.tests('xxx').compare().files()
Q.tests('xxx').compare().file(a='...',b=None,skip=0) #


"""
import json
import numbers

import warnings
import numpy as np
import os
from itertools import chain
from collections import OrderedDict
from ..base_classes.trickitems import NoneItem
from pprint import pformat
from typing import List, Iterator, Set
from datetime import datetime, timedelta
import pandas as pd

from thexp.base_classes.list import llist
from thexp.globals import _FNAME
from thexp.utils.paths import home_dir
from .reader import BoardReader
from .viewer import TestViewer
from ..utils.iters import is_same_type
from ..globals import _BUILTIN_PLUGIN
from .constrain import Constrain, ParamConstrain, MeterConstrain
from ..utils import re

# 虽然好像没什么用但还是设置上了
pd.set_option('display.max_colwidth', 160)
pd.set_option('colheader_justify', 'center')


class Query:

    def tests(self, *items):
        return self.repos().exps().tests(*items)

    def exps(self, *items):
        return self.repos().exps(*items)

    def repos(self, *items):
        if len(items) == 0:
            return ReposQuery(*self.projs_list)
        else:
            return ReposQuery(*self.projs_list)[items]

    @property
    def projs_list(self):
        global_repofn = os.path.join(home_dir(), _FNAME.repo)
        if not os.path.exists(global_repofn):
            res = {}
        else:
            with open(global_repofn, 'r', encoding='utf-8') as r:
                res = json.load(r)

        re_write = False

        repos = []
        projs = []
        for k, v in res.items():  # log_dir(proj level), repopath
            if not os.path.exists(k):  # 不仅需要日志记录，还需要该试验目录确实存在
                re_write = True
                continue

            repos.append(v)
            projs.append(k)

        if re_write:
            with open(global_repofn, 'w', encoding='utf-8') as w:
                json.dump({k: v for k, v in zip(projs, repos)}, w)

        return projs, repos


class ReposQuery:
    """
    以项目为单位的Query，每个项目可能对应多个实验存储路径（一般不主动更改的情况下只有一个）
    """

    def __init__(self, projs: List[str], repos: List[str]):
        self.projs = llist(projs)
        self.repos = llist(repos)

        self.proj_names = [os.path.basename(i) for i in self.projs]

    def __and__(self, other):
        if isinstance(other, ReposQuery):
            combine = set(self.projs) & set(other.projs)
            projs = []
            repos = []
            for proj, repo in chain(zip(self.projs, self.repos), zip(other.projs, other.repos)):
                if proj in combine:
                    combine.remove(proj)
                    projs.append(proj)
                    repos.append(repo)
            return ReposQuery(projs, repos)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, ReposQuery):
            combine = set(self.projs) | set(other.projs)
            projs = []
            repos = []
            for proj, repo in chain(zip(self.projs, self.repos), zip(other.projs, other.repos)):
                if proj in combine:
                    combine.remove(proj)
                    projs.append(proj)
                    repos.append(repo)
            return ReposQuery(projs, repos)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            assert is_same_type(items)
            for item in items:
                if isinstance(item, str):
                    try:
                        idx = self.proj_names.index(item)
                        res.append(idx)
                    except:
                        raise IndexError(item)
                elif isinstance(item, int):
                    res.append(item)
            return ReposQuery(self.projs[res], self.repos[res])  # do not have type error
        elif isinstance(items, int):
            res.append(items)
            return ReposQuery(self.projs[res], self.repos[res])  # do not have type error
        elif isinstance(items, str):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return ReposQuery(self.projs[items], self.repos[items])  # do not have type error

    def __repr__(self):
        return self.df().__repr__()

    __str__ = __repr__

    def __len__(self):
        return len(self.repos)

    def _repr_html_(self):
        with pd.option_context('display.max_colwidth', 0):
            with pd.option_context('colheader_justify', 'center'):
                return self.df()._repr_html_()

    def df(self):
        res = []
        for repo, proj in zip(self.repos, self.projs):
            res.append((os.path.basename(proj), repo))
        return pd.DataFrame(res, columns=['name', 'Repo path'])

    def tests(self, *items):
        return self.exps().tests(*items)

    def exps(self, *items):
        if len(items) == 0:
            viewers = self.to_viewers()
            exp_dirs = []
            for viewer in viewers:
                exp_dirs.extend(viewer.exp_dirs)
            return ExpsQuery(exp_dirs)
        else:
            return self.exps()[items]

    @property
    def empty(self):
        return len(self.repos) == 0

    @property
    def isitem(self):
        return len(self.repos) == 1

    def to_viewer(self):
        if self.isitem:
            from .viewer import ProjViewer
            return ProjViewer(self.projs[0])
        raise ValueError('only one projs contains can be converted to ProjViewer')

    def to_viewers(self):
        from .viewer import ProjViewer
        return [ProjViewer(i) for i in self.projs]

    def delete(self):
        for viewer in self.to_viewers():
            viewer.delete()


class ExpsQuery:
    def __init__(self, exp_dirs):
        self.exp_dirs = llist(exp_dirs)
        self.exp_names = [os.path.basename(i) for i in self.exp_dirs]

    def __and__(self, other):
        if isinstance(other, ExpsQuery):
            combine = set(self.exp_dirs) & set(other.exp_dirs)
            exp_dirs = []
            for exp_dir in chain(self.exp_dirs, other.exp_dirs):
                if exp_dir in combine:
                    combine.remove(exp_dir)
                    exp_dirs.append(exp_dir)
            return ExpsQuery(exp_dirs)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, ExpsQuery):
            combine = set(self.exp_dirs) | set(other.exp_dirs)
            exp_dirs = []
            for exp_dir in chain(self.exp_dirs, other.exp_dirs):
                if exp_dir in combine:
                    combine.remove(exp_dir)
                    exp_dirs.append(exp_dir)
            return ExpsQuery(exp_dirs)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            if len(items) == 0:
                return self
            assert is_same_type(items)
            for i, exp in enumerate(self.exp_names):
                if isinstance(items[0], str) and exp in items:
                    res.append(i)
                elif isinstance(items[0], int) and i in items:
                    res.append(i)
            return ExpsQuery(self.exp_dirs[res])  # do not have type error
        elif isinstance(items, int):
            res.append(items)
            return ExpsQuery(self.exp_dirs[res])  # do not have type error
        elif isinstance(items, str):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return ExpsQuery(self.exp_dirs[items])  # do not have type error
        else:
            raise TypeError(items)

    def __repr__(self):
        if self.empty:
            return '[Empty]'
        return self.df().__repr__()

    __str__ = __repr__

    def __len__(self):
        return len(self.exp_names)

    def _repr_html_(self):
        if self.empty:
            return '<pre>[Empty]</pre>'
        return self.df()._repr_html_()

    def df(self):
        res = []
        names = []
        for viewer in self.to_viewers():
            res.append(viewer.test_names)
            names.append(viewer.name)
        df = pd.DataFrame(res, index=names).T
        return df

    def tests(self, *items):
        if len(items) == 0:
            viewers = self.to_viewers()
            exp_dirs = []
            for viewer in viewers:
                exp_dirs.extend(viewer.test_dirs)
            return TestsQuery(exp_dirs)
        else:
            return self.tests()[items]

    @property
    def empty(self):
        return len(self.exp_dirs) == 0

    @property
    def isitem(self):
        return len(self.exp_dirs) == 1

    def to_viewer(self):
        if self.isitem:
            from .viewer import ExpViewer
            return ExpViewer(self.exp_dirs[0])
        raise ValueError('only one projs contains can be converted to ProjViewer')

    def to_viewers(self):
        from .viewer import ExpViewer
        return [ExpViewer(i) for i in self.exp_dirs]


class BoardQuery():
    def __init__(self, board_readers: List[BoardReader], test_viewers: List[TestViewer]):
        self.board_readers = llist(board_readers)  # type:llist[BoardReader]
        self.test_viewers = llist(test_viewers)  # type:llist[TestViewer]

    def __and__(self, other):
        if isinstance(other, BoardQuery):
            combine = set(self.test_viewers) & set(other.test_viewers)
            readers = []
            test_viewers = []
            for reader, test_viewer in chain(zip(self.board_readers, self.test_viewers),
                                             zip(other.board_readers, other.test_viewers)):
                if test_viewer in combine:
                    combine.remove(test_viewer)
                    readers.append(reader)
                    test_viewers.append(test_viewer)
            return BoardQuery(readers, test_viewers)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, BoardQuery):
            combine = set(self.test_viewers) | set(other.test_viewers)
            readers = []
            test_viewers = []
            for reader, test_viewer in chain(zip(self.board_readers, self.test_viewers),
                                             zip(other.board_readers, other.test_viewers)):
                if test_viewer in combine:
                    combine.remove(test_viewer)
                    readers.append(reader)
                    test_viewers.append(test_viewer)
            return BoardQuery(readers, test_viewers)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            assert is_same_type(items)
            names = [i.name for i in self.test_viewers]
            for item in items:
                if isinstance(item, int):
                    res.append(item)
                elif isinstance(item, str):
                    res.append(names.index(item))
            return BoardQuery(self.board_readers[res], self.test_viewers[res])  # do not have type error
        elif isinstance(items, int):
            res.append(items)
            return BoardQuery(self.board_readers[res], self.test_viewers[res])  # do not have type error
        elif isinstance(items, str):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return BoardQuery(self.board_readers[items], self.test_viewers[items])  # do not have type error

    def __repr__(self):
        return 'BoardQuery({})'.format(pformat(self.test_viewers))

    __str__ = __repr__

    def __len__(self):
        return len(self.board_readers)

    @property
    def param_keys(self):
        res = set()
        for tv in self.test_viewers:  # type:TestViewer
            for k, v in tv.params.inner_dict.walk():
                res.add(k)
        return res

    @property
    def scalar_tags(self) -> Set[str]:
        """返回所有board reader 重合的tags"""
        from functools import reduce
        from operator import and_

        scalars_tags = [set(i.scalars_tags) for i in self.board_readers]
        return reduce(and_, scalars_tags)

    def has_scalar_tags(self, tag):
        res = []
        for i, reader in enumerate(self.board_readers):
            if tag in reader.scalars_tags:
                res.append(i)
        return self[res]

    def values(self, tag, with_step=False):
        res = []
        for reader in self.board_readers:
            try:
                val = reader.get_scalars(tag)
                if not with_step:
                    res.append(val.values)
                else:
                    res.append(val)
            except:
                res.append(None)
        return res

    def parallel_dicts(self, *constrains: Constrain):
        """
        构建 平行图的字典，返回一个字典，包含了每个 tag 对应的所有试验的N个记录

        Args:
            *constrains:
            backend:

        Returns:

        """
        tag_dict = {}

        for constrain in constrains:
            tag = constrain._name
            if isinstance(constrain, ParamConstrain):
                tag_values = list()
                for tv in self.test_viewers:  # type:TestViewer
                    params = tv.params

                    for k, v in params.inner_dict.walk():
                        # check name
                        if k != tag:
                            continue

                        # check constrain conditoin
                        if constrain._constrain != None and not constrain._constrain(v, constrain._value):
                            continue

                        if isinstance(v, (numbers.Number, str)):  # include int, float, bool
                            tag_values.append(v)

                tag_dict[tag] = tag_values

            elif isinstance(constrain, MeterConstrain):
                if constrain._name not in self.scalar_tags:
                    warnings.warn('{} can not found in this query, and will be ignored.'.format(constrain))
                    continue

                constrain_func = constrain._constrain
                values_lis = self.values(tag)
                assert all([len(values) > 0 for values in values_lis])

                if constrain_func is None:  # 自动判断约束类型
                    # 判断方式：tag中 loss 那么取最小值，有 acc 那么取最大值，否则根据初始值和末尾值的大小关系进行判断
                    if 'loss' in tag:
                        constrain_func = np.min
                    elif 'acc' in tag:
                        constrain_func = np.max
                    else:
                        for values in values_lis:
                            if values[0] > values[-1]:
                                constrain_func = np.min
                            else:
                                constrain_func = np.max
                            break

                evalue = [constrain_func(values) for values in values_lis]

                tag_dict[tag] = evalue

        return tag_dict

    def line(self, tag, backend='matplotlib'):
        """plot values of the tag of the tests selected"""
        from .charts import Curve
        figure = {}
        for bd, test_viewer in zip(self.board_readers, self.test_viewers):
            test_name = test_viewer.name

            scalars = bd.get_scalars(tag)
            figure[test_name] = {
                'name': test_name,
                'x': scalars.steps,
                'y': scalars.values,
            }
        curve = Curve(figure, title=tag)
        plot_func = getattr(curve, backend, None)
        if plot_func is None:
            raise NotImplementedError(backend)
        else:
            return plot_func()

    def tvalues(self, tags: List[str], with_step=False):
        """get tags of one test"""
        assert len(self.board_readers) == 1
        res = []
        for tag in tags:
            try:
                val = self.board_readers[0].get_scalars(tag)
                if not with_step:
                    res.append(val.values)
                else:
                    res.append(val)
            except:
                res.append(None)
        return res

    def tline(self, tags: List[str], backend='matplotlib'):
        """line of multi-tags of one test"""
        assert len(self.board_readers) == 1
        from .charts import Curve
        figure = {}
        for tag in tags:
            scalars = self.board_readers[0].get_scalars(tag)
            figure[tag] = {
                'name': tag,
                'x': scalars.steps,
                'y': scalars.values,
            }
        curve = Curve(figure, title=self.test_viewers[0].name)
        plot_func = getattr(curve, backend, None)
        if plot_func is None:
            raise NotImplementedError(backend)
        else:
            return plot_func()

    def parallel(self, *constrains: Constrain, backend='matplotlib'):
        """draw parallel with the constrains to compare tests"""
        tag_dict = self.parallel_dicts(*constrains)
        from .charts import Parallel
        parallel = Parallel([i.name for i in self.test_viewers], tag_dict)

        plot_func = getattr(parallel, backend, None)
        if plot_func is None:
            raise NotImplementedError(backend)
        else:
            return plot_func()

    def summary(self):
        from collections import defaultdict
        res = defaultdict(list)
        for br, tv in zip(self.board_readers, self.test_viewers):  # type:BoardReader,TestViewer
            for tag in br.scalars_tags:
                res[tag].append(tv.name)

        print('Scalar tags:')
        for k, v in res.items():
            if len(v) == len(self.board_readers):
                print('{} (all)'.format(k))
            else:
                print('{} ({})'.format(k, v))


class TestsQuery:
    """
    支持模糊匹配
    排序
    ts.sort_by_start_time(...)

    切片
    ts[:4]

    查找
    ts.INT[32]
    ts["0032"]
    ts.has_board()
    ts.success()

    标记
    ts.mark("mark")
    ts.unmark("mark")
    """
    def __init__(self, test_dirs: List[str]):
        self.test_dirs = llist(test_dirs)
        self.test_names = [os.path.basename(i) for i in self.test_dirs]
        self._int_match = False

    def __and__(self, other):
        if isinstance(other, TestsQuery):
            combine = set(self.test_names) & set(other.test_names)
            test_dirs = []
            for test_dir, test_name in chain(zip(self.test_dirs, self.test_names),
                                             zip(other.test_dirs, other.test_names)):
                if test_name in combine:
                    combine.remove(test_name)
                    test_dirs.append(test_dir)
            return TestsQuery(test_dirs)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, TestsQuery):
            combine = set(self.test_names) | set(other.test_names)
            test_dirs = []
            for test_dir, test_name in chain(zip(self.test_dirs, self.test_names),
                                             zip(other.test_dirs, other.test_names)):
                if test_name in combine:
                    combine.remove(test_name)
                    test_dirs.append(test_dir)
            return TestsQuery(test_dirs)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            assert is_same_type(items)
            for item in items:
                if self._int_match:
                    item = str(item)

                if isinstance(item, str):

                    if len(item) < 13:  # not a specific test name
                        match = re.compile(item)
                        for idx, _name in enumerate(self.test_names):
                            if re.search(match, _name) is not None:
                                res.append(idx)
                    else:
                        try:
                            idx = self.test_names.index(item)
                            res.append(idx)
                        except:
                            raise IndexError(item)
                elif isinstance(item, int):
                    res.append(item)
            return TestsQuery(self.test_dirs[res])  # do not have type error
        elif isinstance(items, (int, str)):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return TestsQuery(self.test_dirs[items])  # do not have type error

    def __repr__(self):
        if self.empty:
            return '[Empty]'
        return self.df().__repr__()

    def _repr_html_(self):
        if self.empty:
            return '<pre>[Empty]</pre>'
        return self.df()._repr_html_()

    __str__ = __repr__

    def __len__(self):
        return len(self.test_names)

    def __iter__(self):
        for dir in self.test_dirs:
            yield self[dir]

    def _copy(self):
        return TestsQuery(self.test_dirs)

    """链式调用糖"""

    @property
    def INT(self):
        """int 切片不是按顺序，而是找实验编号"""
        res = self._copy()
        res._int_match = True
        return res

    @property
    def TIME(self):
        """
        ts.TIME.left().right().sorted().done()
        Returns:
            TODO
        """
        raise NotImplementedError()

    @property
    def SORTED(self):
        """
        ts.SORTED.runtime(descending=True)
        ts.SORTED.starttime()
        ts.SORTED.endtime()
        Returns:
            TODO
        """
        raise NotImplementedError()

    @property
    def empty(self):
        return len(self.test_dirs) == 0

    @property
    def isitem(self):
        return len(self.test_dirs) == 1

    def sort_by_start_time(self, descending=False):
        _, test_dirs = zip(
            *sorted(
                zip(self.to_viewers(), self.test_dirs),
                key=lambda x: x.to_viewer().start_time,
                reverse=descending
            )
        )
        return self[test_dirs]

    def df(self):
        df = pd.DataFrame([viewer.df_info() for viewer in self.to_viewers()],
                          columns=self[0].to_viewer().df_columns(),
                          index=self.test_names)
        return df

    def params_df(self):
        params_lis = [vw.params for vw in self.to_viewers()]
        res = []
        for params in params_lis:
            dic = OrderedDict()
            for k, v in params.inner_dict.walk():
                dic[k] = v
            res.append(dic)

        df = pd.DataFrame(res, index=[vw.name for vw in self.to_viewers()]).T
        return df

    def boards(self):
        from thexp.base_classes.errors import NoneWarning
        viewers = []
        boards = []
        for vw in self.to_viewers():
            if vw.board_reader is None:
                warnings.warn('{} have no board, they will be removed.'.format(vw.name), NoneWarning)
                continue
            viewers.append(vw)
            boards.append(vw.board_reader)
        return BoardQuery(boards, viewers)

    def to_viewers(self):
        """convert testquery to  testviewer list"""
        from .viewer import TestViewer
        return [TestViewer(i) for i in self.test_dirs]

    def to_viewer(self):
        """convert testquery to single testviewer if there is only one test in queryset"""
        if self.isitem:
            from .viewer import TestViewer
            return TestViewer(self.test_dirs[0])
        raise ValueError('only one test_dirs contains can be converted to TestViewer')

    """filter tests"""

    def first(self):
        return self[0]

    def head(self, num=5):
        return self[:5]

    def last(self):
        return self[-1]

    def tail(self, num=5):
        return self[-5:]

    def time_range(self, left_time=None, right_time=None):
        """
        筛选 start_time 位于某时间区间内的 test
        Args:
            left_time:
            right_time:

        Returns:

        """
        if left_time is None and right_time is None:
            return self

        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if left_time is None:
                if viewer.start_time < right_time:
                    res.append(i)
            elif right_time is None:
                if viewer.start_time > left_time:
                    res.append(i)
            else:
                if viewer.start_time > left_time and viewer.start_time < left_time:
                    res.append(i)

        return self[res]

    def in_time(self,
                minutes=0,
                hours=0,
                days=0,
                weeks=0,
                seconds=0):
        delta = timedelta(minutes=minutes, days=days, hours=hours, weeks=weeks, seconds=seconds)
        return self.time_range(left_time=datetime.now() - delta)

    def success(self):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.success_exit:
                res.append(i)
        return self[res]

    def failed(self):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if not viewer.success_exit:
                res.append(i)
        return self[res]

    def has_tag(self, tag: str, toggle=True):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.has_tag(tag):
                if toggle:
                    res.append(i)
            else:
                if not toggle:
                    res.append(i)
        return self[res]

    def has_plugin(self, plugin, toggle=True):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.has_plugin(plugin):
                if toggle:
                    res.append(i)
            else:
                if not toggle:
                    res.append(i)

        return self[res]

    def has_board(self, toggle=True):
        return self.has_plugin(_BUILTIN_PLUGIN.writer, toggle)

    def has_log(self, toggle=True):
        return self.has_plugin(_BUILTIN_PLUGIN.logger, toggle)

    def has_saver(self, toggle=True):
        return self.has_plugin(_BUILTIN_PLUGIN.saver, toggle)

    def has_params(self, toggle=True):
        return self.has_plugin(_BUILTIN_PLUGIN.params, toggle)

    def has_trainer(self, toggle=True):
        return self.has_plugin(_BUILTIN_PLUGIN.trainer, toggle)

    def has_dir(self, dir: str, toggle=True):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.has_dir(dir):
                if toggle: res.append(i)
            else:
                if not toggle: res.append(i)
        return self[res]

    def has_state(self, state_name: str, toggle=True):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.has_state(state_name):
                if toggle: res.append(i)
            else:
                if not toggle: res.append(i)
        return self[res]

    def filter_params(self, *constrains: ParamConstrain):
        """
        Query test with param constrains.

        Args:
            *constrains: a ParamConstrain instance.

        Returns:

        """
        if len(constrains) == 0:
            return self

        res = []
        params_lis = [vw.params for vw in self.to_viewers()]
        for i, params in enumerate(params_lis):
            feasible = True
            for constrain in constrains:
                if params is None:
                    feasible = False
                    break

                left = params.get(constrain._name, NoneItem())
                if isinstance(left, NoneItem) and not constrain._allow_none:
                    feasible = False
                    break

                right = constrain._value
                if not constrain._constrain(left, right):
                    feasible = False
                    break
            if feasible:
                res.append(i)

        return self[res]

    """update test state"""

    def delete(self):
        for viewer in self.to_viewers():
            viewer.delete()

    def delete_modules(self):
        for viewer in self.to_viewers():
            viewer.delete_modules()

    def delete_keypoints(self):
        for viewer in self.to_viewers():
            viewer.delete_keypoints()

    def delete_checkpoints(self):
        for viewer in self.to_viewers():
            viewer.delete_checkpoints()

    def toggle_state(self, state_name: str, toggle=None):
        for viewer in self.to_viewers():
            viewer.toggle_state(state_name, toggle)

    def mark(self, state_name: str):
        self.toggle_state(state_name, True)

    def unmark(self, state_name: str):
        self.toggle_state(state_name, False)

    def hide(self):
        for viewer in self.to_viewers():
            viewer.hide()

    def show(self):
        for viewer in self.to_viewers():
            viewer.show()

    def fav(self):
        for viewer in self.to_viewers():
            viewer.fav()

    def unfav(self):
        for viewer in self.to_viewers():
            viewer.unfav()


Q = Query()
