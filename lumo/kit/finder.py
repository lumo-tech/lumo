"""
Help you find the experiments you have done.
"""

# from git import Commit, Repo
#
#
# class TestProp():
#     @property
#     def command(self):
#         raise NotImplementedError()
#
#     @property
#     def exp_name(self):
#         raise NotImplementedError()
#
#     @property
#     def exp_root(self):
#         raise NotImplementedError()
#
#     @property
#     def project_name(self):
#         raise NotImplementedError()
#
#     @property
#     def repo_root(self):
#         raise NotImplementedError()
#
#     @property
#     def repo_hash(self):
#         raise NotImplementedError()
#
#     @property
#     def start_time(self):
#         raise NotImplementedError()
#
#     @property
#     def end_time(self):
#         raise NotImplementedError()
#
#     @property
#     def end_code(self):
#         raise NotImplementedError()
#
#     @property
#     def grepo(self) -> Repo:
#         raise NotImplementedError()
#
#     @property
#     def gcommit(self) -> Commit:
#         raise NotImplementedError()
#
#     @property
#     def commit_hash(self) -> str:
#         raise NotImplementedError()
#
#     @property
#     @lru_cache(1)
#     def lines(self) -> Dict[str, Dict]:
#         raise NotImplementedError()
#
#     @property
#     @lru_cache(1)
#     def jsons(self):
#         raise NotImplementedError()
#
#     @property
#     def pkl_keys(self):
#         raise NotImplementedError()
#
#     @property
#     def uuid(self):
#         raise NotImplementedError()
#
#     @property
#     def largest_epoch(self):
#         raise NotImplementedError()
#
#
# @dataclass(unsafe_hash=True)
# class Test(TestProp):
#     tid: str
#     root: str
#
#     @property
#     def command(self):
#         exe = self.jsons[EXP.EXECUTE]['exec_bin']
#         argv = self.jsons[EXP.EXECUTE]['exec_argv']
#         return ' '.join([exe, *argv])
#
#     @property
#     def exp_name(self):
#         return os.path.basename(self.exp_root)
#
#     @property
#     def exp_root(self):
#         return os.path.dirname(self.root)
#
#     @property
#     def project_name(self):
#         return os.path.basename(self.repo_root)
#
#     @property
#     def repo_root(self):
#         return self.jsons[EXP.PROJECT]['root']
#
#     @property
#     def repo_hash(self):
#         return self.jsons[EXP.PROJECT]['hash']
#
#     @property
#     def start_time(self):
#         val = self.jsons[EXP.STATE].get('start', None)
#         if val is not None:
#             val = date_from_str(val)
#         return val
#
#     @property
#     def end_time(self):
#         val = self.jsons[EXP.STATE].get('end', None)
#         if val is not None:
#             val = date_from_str(val)
#         return val
#
#     @property
#     def end_code(self):
#         return self.jsons[EXP.STATE].get('end_code', 1)
#
#     @property
#     def grepo(self) -> Repo:
#         raise NotImplementedError()
#         # repo = load_repo(self.repo_root)
#         # return repo
#
#     @property
#     def gcommit(self) -> Commit:
#         from gitdb.util import hex_to_bin
#         return Commit(self.grepo, hex_to_bin(self.commit_hash))
#
#     @property
#     def commit_hash(self) -> str:
#         """None if nocommit in this test, or commit hash will be returned"""
#         return self.jsons.get(EXP.GIT, {}).get('commit', None)
#
#     @property
#     def lines(self) -> Dict[str, str]:
#         dir_ = os.path.join(self.root, FN.D_LINE)
#         if not os.path.exists(dir_):
#             return attr()
#         fs_ = os.listdir(dir_)
#         fs_ = [i for i in fs_ if i.endswith(FN.SUFFIX.D_LINE)]
#         res = attr()
#         for f in fs_:
#             k = os.path.splitext(f)[0]
#             v = io.load_string(os.path.join(dir_, f))
#             if v is None:
#                 os.remove(os.path.join(dir_, f))
#                 continue
#             res[k] = v
#         return res
#
#     @property
#     def jsons(self):
#         dir_ = os.path.join(self.root, FN.D_JSON)
#         if not os.path.exists(dir_):
#             return attr()
#         fs_ = os.listdir(dir_)
#         fs_ = [i for i in fs_ if i.endswith('.json')]
#         res = attr()
#         for f in fs_:
#             k = os.path.splitext(f)[0]
#             v = io.load_json(os.path.join(dir_, f))
#             if v is None:
#                 continue
#             res[k] = v
#         return res
#
#     @property
#     @lru_cache(1)
#     def pkl_keys(self):
#         dir_ = os.path.join(self.root, FN.D_JSON)
#         if not os.path.exists(dir_):
#             return set()
#
#         fs_ = os.listdir(dir_)
#         fs_ = [i for i in fs_ if i.endswith('.pkl')]
#         res = set()
#         for f in fs_:
#             k = os.path.splitext(f)[0]
#             res.add(k)
#         return res
#
#     @property
#     def uuid(self):
#         return self.lines.get(EXP.UUID, '')
#
#     @property
#     def largest_epoch(self):
#         return self.jsons.get(EXP.TRAINER, {}).get('epoch', 0)
#
#     def pkl(self, key):
#         """
#
#         Args:
#             key:
#
#         Returns:
#
#         """
#         dir_ = os.path.join(self.root, FN.D_JSON)
#         f = os.path.join(dir_, f"{key}.pkl")
#         return io.load_state_dict(f)
#
#
# class Condition():
#     def filter(self, test: Test) -> bool:
#         raise NotImplementedError()
#
#     def __call__(self, tests: List[Test]):
#         return [i for i in tests if self.filter(i)]
#
#
# class LambdaCondition(Condition):
#     def __init__(self, func):
#         self.func = func
#
#     def filter(self, test: Test) -> bool:
#         return self.func(test)
#
#
# class Identity(Condition):
#
#     def filter(self, test: Test) -> bool:
#         return True
#
#
# class Or(Condition):
#     def __init__(self, *conditions: Condition):
#         self.conditions = list(conditions)
#
#     def add(self, condition: Condition):
#         self.conditions.append(condition)
#
#     def filter(self, test: Test) -> bool:
#         return any([i.filter(test) for i in self.conditions])
#
#
# class And(Or):
#     def filter(self, test: Test) -> bool:
#         return all([i.filter(test) for i in self.conditions])
#
#
# class Query:
#     def __init__(self):
#         self.conditions = And()
#
#     def success(self):
#         self.conditions.add(LambdaCondition(lambda x: x.end_code == 0))
#         return self
#
#     def failed(self):
#         self.conditions.add(LambdaCondition(lambda x: x.end_code != 0))
#         return self
#
#     def has_info(self, key):
#         self.conditions.add(LambdaCondition(lambda x: x.jsons.get(key, None) is not None))
#         return self
#
#     def has_line(self, key):
#         self.conditions.add(LambdaCondition(lambda x: x.lines.get(key, None) is not None))
#         return self
#
#     def in_time(self, start=None, end=None):
#         if start is not None:
#             self.conditions.add(LambdaCondition(lambda x: x.start_time > start is not None))
#         if end is not None:
#             self.conditions.add(LambdaCondition(lambda x: x.end_time > end is not None))
#         return self
#
#     def train_longer_than(self, epoch=0):
#         self.conditions.add(LambdaCondition(lambda x: x.largest_epoch > epoch))
#         return self
#
#     def train_shorter_than(self, epoch=1):
#         self.conditions.add(LambdaCondition(lambda x: x.largest_epoch <= epoch))
#         return self
#
#     def in_repo(self, name=None, path=None, full=False):
#         if name is not None:
#             if full:
#                 self.conditions.add(LambdaCondition(lambda x: name == x.project_name))  # type:Test
#             else:
#                 self.conditions.add(LambdaCondition(lambda x: name in x.project_name))  # type:Test
#         elif path is not None:
#             self.conditions.add(LambdaCondition(lambda x: compare_path(path, x.repo_root)))  # type:Test
#         else:
#             path = git_dir(path)
#             if path is not None:
#                 return self.in_repo(path=path)
#             else:
#                 return Identity()
#
#         return self
#
#     def build(self):
#         return self.conditions
#
#
# class Finder:
#     def __init__(self):
#         fn = os.path.join(libhome(), FN.REPOSJS)
#         if os.path.exists(fn):
#             res = io.load_json(fn)
#         else:
#             res = {}
#         self.meta = res
#         self._test_dirs = []
#         self._tests = Tests()
#         self.refresh()
#
#     def refresh(self):
#         fn = os.path.join(libhome(), FN.TESTLOG)
#         if not os.path.exists(fn):
#             return
#         res = io.load_string(fn)
#         if res is None:
#             res = ''
#         res = res.split('\n')
#
#         self._test_dirs = list(OrderedDict.fromkeys(
#             filter(lambda x: os.path.exists(x.strip()), res)
#         ).keys())
#         self._test_dirs = sorted(self._test_dirs, key=lambda x: os.stat(x).st_atime)
#
#         self._tests = Tests(Test(os.path.basename(i), i) for i in self._test_dirs)
#
#     def tests(self, *conditions, nonflag=None):
#         """can be found in each expdir"""
#         tests = self._tests
#         for condition in conditions:
#             tests = condition(tests)
#         if len(tests) == 0:
#             return nonflag
#         return Tests(tests)
#
#     def lastest(self) -> Test:
#         return self._tests[-1]
#
#     def n_lastest(self, n) -> List[Test]:
#         return self._tests[-n:]
#
#
# class Tests(TestProp, List[Test]):
#
#     def _to_df(self, value: list, column: str):
#         import pandas as pd
#         df = pd.DataFrame()
#         df[column] = value
#         df.index = self.names
#         return df
#
#     @property
#     def command(self):
#         return self._to_df([i.command for i in self], 'command')
#
#     @property
#     def exp_name(self):
#         return self._to_df([i.exp_name for i in self], 'exp_name')
#
#     @property
#     def exp_root(self):
#         return self._to_df([i.exp_root for i in self], 'exp_root')
#
#     @property
#     def project_name(self):
#         return self._to_df([i.project_name for i in self], 'project_name')
#
#     @property
#     def repo_root(self):
#         return self._to_df([i.repo_root for i in self], 'repo_root')
#
#     @property
#     def repo_hash(self):
#         return self._to_df([i.repo_hash for i in self], 'repo_hash')
#
#     @property
#     def start_time(self):
#         return self._to_df([i.start_time for i in self], 'start_time')
#
#     @property
#     def end_time(self):
#         return self._to_df([i.end_time for i in self], 'end_time')
#
#     @property
#     def end_code(self):
#         return self._to_df([i.end_code for i in self], 'end_code')
#
#     @property
#     def grepo(self):
#         return self._to_df([i.grepo for i in self], 'grepo')
#
#     @property
#     def gcommit(self):
#         return self._to_df([i.gcommit for i in self], 'gcommit')
#
#     @property
#     def commit_hash(self):
#         return self._to_df([i.commit_hash for i in self], 'commit_hash')
#
#     @property
#     def lines(self):
#         return [i.lines for i in self]
#
#     @property
#     def jsons(self):
#         return [i.jsons for i in self]
#
#     @property
#     def pkl_keys(self):
#         return [i.pkl_keys for i in self]
#
#     @property
#     def names(self):
#         return [i.tid for i in self]
#
#     @property
#     def uuid(self):
#         return self._to_df([i.uuid for i in self], 'uuid')
#
#     @property
#     def latest(self):
#         return self[-1]
#
#     @property
#     def largest_epoch(self):
#         return self._to_df([i.largest_epoch for i in self], 'largest_epoch')
#
#     @property
#     def root(self):
#         return self._to_df([i.root for i in self], 'root')
#
#     def delete(self):
#         for i in self:
#             shutil.rmtree(i.root)
#
#
# F = Finder()
# Q = Query()
