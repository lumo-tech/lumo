"""
Help you find the experiments you have done.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

from git import Commit, Repo

from lumo.utils import safe_io as io
from lumo.utils.dates import date_from_str
from lumo.utils.keys import FN, EXP
from lumo.utils.paths import home_dir, compare_path
from lumo.utils.repository import load_repo


@dataclass()
class Test():
    name: str
    root: str

    @property
    def command(self):
        exe = self.lines[EXP.EXECUTE]['exec_bin']
        argv = self.lines[EXP.EXECUTE]['exec_argv']
        return ' '.join([exe, *argv])

    @property
    def exp_name(self):
        return os.path.basename(self.exp_root)

    @property
    def exp_root(self):
        return os.path.dirname(self.root)

    @property
    def project_name(self):
        return os.path.basename(self.repo_root)

    @property
    def repo_root(self):
        return self.lines[EXP.PROJECT]['root']

    @property
    def repo_hash(self):
        return self.lines[EXP.PROJECT]['hash']

    @property
    def start_time(self):
        return date_from_str(self.lines[EXP.STATE]['start'])

    @property
    def end_time(self):
        return date_from_str(self.lines[EXP.STATE].get('end', None))

    @property
    def end_code(self):
        return self.lines[EXP.STATE].get('end_code', 1)

    @property
    def grepo(self) -> Repo:
        repo = load_repo(self.repo_root)
        return repo

    @property
    def gcommit(self) -> Commit:
        from gitdb.util import hex_to_bin
        return Commit(self.grepo, hex_to_bin(self.commit_hash))

    @property
    def commit_hash(self) -> str:
        """None if nocommit in this test, or commit hash will be returned"""
        return self.lines.get(EXP.GIT, {}).get('commit', None)

    @property
    @lru_cache(1)
    def lines(self) -> Dict[str, Dict]:
        dir_ = os.path.join(self.root, FN.D_LINE)
        if not os.path.exists(dir_):
            return {}
        fs_ = os.listdir(dir_)
        fs_ = [i for i in fs_ if i.endswith('.')]
        res = {}
        for f in fs_:
            k = os.path.splitext(f)[0]
            v = io.load_string(os.path.join(dir_, f))
            res[k] = v
        return res

    @property
    @lru_cache(1)
    def jsons(self):
        dir_ = os.path.join(self.root, FN.D_JSON)
        if not os.path.exists(dir_):
            return {}
        fs_ = os.listdir(dir_)
        fs_ = [i for i in fs_ if i.endswith('.json')]
        res = {}
        for f in fs_:
            k = os.path.splitext(f)[0]
            v = io.load_json(os.path.join(dir_, f))
            res[k] = v
        return res

    @property
    @lru_cache(1)
    def pkl_keys(self):
        dir_ = os.path.join(self.root, FN.D_JSON)
        if not os.path.exists(dir_):
            return set()

        fs_ = os.listdir(dir_)
        fs_ = [i for i in fs_ if i.endswith('.pkl')]
        res = set()
        for f in fs_:
            k = os.path.splitext(f)[0]
            res.add(k)
        return res

    def pkl(self, key):
        """

        Args:
            key:

        Returns:

        """
        dir_ = os.path.join(self.root, FN.D_JSON)
        f = os.path.join(dir_, f"{key}.pkl")
        return io.load_state_dict(f)


class Condition():
    def filter(self, test: Test) -> bool:
        raise NotImplementedError()

    def __call__(self, tests: List[Test]):
        return [i for i in tests if self.filter(i)]


class LambdaCondition(Condition):
    def __init__(self, func):
        self.func = func

    def filter(self, test: Test) -> bool:
        return self.func(test)


class Or(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = list(conditions)

    def add(self, condition: Condition):
        self.conditions.append(condition)

    def filter(self, test: Test) -> bool:
        return any([i.filter(test) for i in self.conditions])


class And(Or):
    def filter(self, test: Test) -> bool:
        return all([i.filter(test) for i in self.conditions])


class Query:
    def __init__(self):
        self.conditions = And()

    def success(self):
        self.conditions.add(LambdaCondition(lambda x: x.end_code == 0))
        return self

    def failed(self):
        self.conditions.add(LambdaCondition(lambda x: x.end_code != 0))
        return self

    def has_info(self, key):
        self.conditions.add(LambdaCondition(lambda x: x.jsons.get('key', None) is not None))
        return self

    def in_time(self, start=None, end=None):
        if start is not None:
            self.conditions.add(LambdaCondition(lambda x: x.start_time > start is not None))
        if end is not None:
            self.conditions.add(LambdaCondition(lambda x: x.end_time > end is not None))
        return self

    def in_repo(self, name=None, path=None, full=False):
        if name is not None:
            if full:
                self.conditions.add(LambdaCondition(lambda x: name == x.project_name))  # type:Test
            else:
                self.conditions.add(LambdaCondition(lambda x: name in x.project_name))  # type:Test
        elif path is not None:
            self.conditions.add(LambdaCondition(lambda x: compare_path(path, x.repo_root)))  # type:Test
        return self


class Finder:
    def __init__(self):
        fn = os.path.join(home_dir(), FN.REPOSJS)
        if os.path.exists(fn):
            res = io.load_json(fn)
        else:
            res = {}
        self.meta = res
        self._test_dirs = []
        self._tests = Filter()
        self.refresh()

    def refresh(self):
        fn = os.path.join(home_dir(), FN.TESTLOG)
        with open(fn, 'r', encoding='utf-8') as r:
            self._test_dirs = list(OrderedDict.fromkeys(i.strip() for i in r if os.path.exists(i.strip())).keys())

        self._tests = Filter(Test(os.path.basename(i), i) for i in self._test_dirs)

    def tests(self, *conditions):
        """can be found in each expdir"""
        tests = self._tests
        for condition in conditions:
            tests = condition(tests)

        return self._tests

    def lastest(self) -> Test:
        return self._tests[-1]

    def n_lastest(self, n) -> List[Test]:
        return self._tests[-n:]


class Filter(list):
    pass


F = Finder()
