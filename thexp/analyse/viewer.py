"""
对各不同级别的实验日志存储信息的查看
"""
import json
import os
import pprint
import shutil
from functools import lru_cache
from typing import Dict

from thexp import Experiment
from thexp.base_classes.attr import attr
from thexp.globals import _INFOJ, _FNAME, _BUILTIN_PLUGIN, _GITKEY, _TEST_BUILTIN_STATE, _PLUGIN_DIRNAME, _PLUGIN_KEY
from .reader import BoardReader
from ..utils.dates import date_from_str


class _Viewer():
    def delete(self):

        dir_ = self.root
        if dir_ is None:
            return
        shutil.rmtree(dir_)

        from itertools import chain
        for attr_name in chain(dir(self.__class__), dir(self)):
            if attr_name.startswith('__'):
                continue
            try:
                attr = getattr(self, attr_name)
            except:
                attr = None
            try:
                if callable(attr):
                    setattr(self, attr_name, lambda *args, **kwargs: 'NoneType')
                else:
                    setattr(self, attr_name, None)
            except:
                pass
        self.__str__ = lambda: ''
        self.__repr__ = lambda: ''

    @property
    def root(self):
        raise NotImplementedError()

    def tree(self, depth: int = 0):
        raise NotImplementedError()

    def summary(self):
        return pprint.pprint(self.tree(0))


class ProjViewer(_Viewer):
    """
    以项目为单位的实验浏览类，用于查看所有 未删除 的 exp 对应的方法
    """

    def __init__(self, proj_dir: str):
        self._proj_dir = proj_dir
        self._repo = None
        self._repo_info = None

    def __iter__(self):
        for exp_viewer in self.exp_viewers:
            for test_viewer in exp_viewer:
                yield test_viewer

    def __repr__(self):
        return "ProjViewer(exps=[{}])".format(self.exp_names)

    @property
    def name(self):
        return os.path.basename(self.root)

    @property
    def root(self):
        return self._proj_dir

    @property
    def hash(self):
        return self.name.split('.')[-1]

    @property
    def repopath(self):
        with open(os.path.join(self.root, _FNAME.repopath), 'r', encoding='utf-8') as r:
            return r.readline().strip()

    @property
    def repo(self):
        if self._repo is None:
            from git import Repo
            self._repo = Repo(self.repopath)
        return self._repo

    @property
    def exp_dirs(self):
        fn = [os.path.join(self.root, i) for i in os.listdir(self.root)]
        fn = [f for f in fn if os.path.exists(os.path.join(f, _FNAME.repopath))]
        return fn

    @property
    def exp_names(self):
        return [os.path.basename(i) for i in self.exp_dirs]

    @property
    def exp_viewers(self):
        return [ExpViewer(i) for i in self.exp_dirs]

    def tree(self, depth=None):
        if depth == 0:
            return {
                "repopath": self.repopath,
                "name": self.name,
                "root": self.root,
                "exps": [k for k in self.exp_names]
            }
        else:
            if depth is not None:
                depth -= 1

            return {
                "repopath": self.repopath,
                "name": self.name,
                "root": self.root,
                "exps": {
                    os.path.basename(k): ExpViewer(k).tree(depth)
                    for k in self.exp_dirs
                }
            }

    def summary(self):
        print('Repo : {}'.format(self.repopath))
        print('Project Dir : {}'.format(self.root))
        print('Experiments :')
        for k in self.exp_dirs:
            print('  {}'.format(k))


class ExpViewer(_Viewer):
    """
    :param trainer_cls: cls of trainer or str
    """

    def __init__(self, exp_dir: str):
        self._exp_dir = exp_dir
        self._repo = None
        self._name = os.path.basename(self._exp_dir)

    def __iter__(self):
        for f in self.test_dirs:
            yield TestViewer(f)

    def __repr__(self):
        return "ExpViewer({} tests)".format(len(self.test_dirs))

    @property
    def name(self):
        return self._name

    @property
    def root(self):
        return self._exp_dir

    @property
    def repopath(self):
        with open(os.path.join(self.root, _FNAME.repopath), encoding='utf-8') as r:
            return r.readline().strip()

    @property
    def repo(self):
        if self._repo is None:
            from git import Repo
            self._repo = Repo(self.repopath)
        return self._repo

    @property
    def test_names(self):
        return [os.path.basename(i) for i in self.test_dirs]

    @property
    def test_dirs(self):
        if not os.path.exists(self.root):
            return []
        fs = [os.path.join(self.root, f) for f in os.listdir(self.root)]
        fs = [f for f in fs if os.path.exists(os.path.join(f, _FNAME.info))]
        fs = sorted(fs, key=lambda f: os.path.getmtime(f))
        return fs

    @property
    def test_viewers(self):
        """默认返回所有可见的 试验 的 TestViewer 对象"""
        return list(iter(self))

    @property
    def proj_name(self):
        return os.path.basename(self.proj_dir)

    @property
    def proj_dir(self):
        return os.path.dirname(self.root)

    @property
    def projviewer(self):
        return ProjViewer(self.proj_dir)

    def tree(self, depth=None):
        if depth == 0:
            return {
                "repopath": self.repopath,
                "name": self.name,
                "root": self.root,
                'tests': self.test_names
            }
        return {
            "repopath": self.repopath,
            "name": self.name,
            "root": self.root,
            'tests': {
                os.path.basename(k): TestViewer(k).tree() for k in self.test_dirs
            }
        }


class TestViewer(_Viewer):

    def __init__(self, test_dir: str):
        self._test_dir = test_dir
        self._repo = None
        self._commit = None
        self.count = int(os.path.basename(test_dir).split(".")[0])
        self._json_info = None

    def __repr__(self):
        return "Test({})".format(self.name)

    """base variables"""

    @property
    def name(self) -> str:
        return self.json_info[_INFOJ.test_name]

    @property
    def root(self):
        return self._test_dir

    @property
    def repopath(self) -> str:
        return self.json_info[_INFOJ.repo]

    """git varialbes"""

    @property
    def repo(self):
        if self._repo is None:
            from git import Repo
            self._repo = Repo(self.repopath)
        return self._repo

    @property
    def commit(self):
        if self._commit is None:
            from gitdb.util import hex_to_bin
            from git import Commit
            self._commit = Commit(self.repo, hex_to_bin(self.commit_hash))
        return self._commit

    """parent varialbes"""

    @property
    def exp_dir(self):
        return os.path.dirname(self.root)

    @property
    def expviewer(self):
        return ExpViewer(self.exp_dir)

    @property
    def test_order_count(self) -> int:
        return int(self.name.split('.')[0])

    """file infomation"""

    @property
    def fn_global_info(self) -> str:
        return os.path.join(self.root, _FNAME.info)

    """test base infomation"""

    @property
    def json_info(self) -> dict:
        if self._json_info is None:
            if not os.path.exists(self.fn_global_info):
                self._json_info = attr()
                res = self._json_info
            with open(self.fn_global_info, encoding='utf-8') as r:
                try:
                    self._json_info = attr(json.load(r))
                    res = self._json_info
                except json.JSONDecodeError as je:
                    print('Error when decoding test {}, path = {}'.format(self.name, self.root))
                    res = attr()

            res["test_dir"] = self.root
            res['visible'] = self.visible
            res['fav'] = self.isfav
            res['states'] = self.states
            return res.jsonify()
        return self._json_info.jsonify()

    @property
    def argv(self):
        return self.json_info[_INFOJ.argv]

    @property
    def commit_hash(self) -> str:
        return self.json_info[_INFOJ.commit_hash]

    @property
    def short_hash(self) -> str:
        return self.json_info[_INFOJ.short_hash]

    @property
    def success_exit(self) -> bool:
        return _INFOJ.end_code in self.json_info and self.json_info[_INFOJ.end_code] == 0

    @property
    def start_time(self):
        return date_from_str(self.json_info[_INFOJ.start_time], self.json_info[_INFOJ.time_fmt])

    @property
    def end_time(self):
        return date_from_str(self.json_info[_INFOJ.end_time], self.json_info[_INFOJ.time_fmt])

    @property
    def duration(self):
        return self.end_time - self.start_time

    """base plugins/tags operation"""

    @property
    def plugins(self) -> dict:
        dic = self.json_info[_INFOJ.plugins]
        for tag, plugins in self.json_info[_INFOJ.tags].items():
            for plugin in plugins:
                if plugin not in dic:
                    dic[plugin] = {
                        _INFOJ._tags: []
                    }
                else:
                    lis = dic[plugin].setdefault(_INFOJ._tags, [])
                    lis.append(tag)
        return dic

    @property
    def tags(self):
        dic = dict()
        for tag, plugins in self.json_info[_INFOJ.tags].items():
            if len(plugins) == 0:
                plugins = [_INFOJ._]
            for plugin in plugins:
                lis = dic.setdefault(plugin, [])
                lis.append(tag)
        return dic

    def has_tag(self, tag: str):
        return _INFOJ.tags in self.json_info and tag in self.json_info[_INFOJ.tags]

    def has_dir(self, dir: str):
        return _INFOJ.dirs in self.json_info and \
               dir in self.json_info[_INFOJ.dirs] and \
               os.path.exists(os.path.join(self.root, dir))

    def has_plugin(self, plugin: str):
        return _INFOJ.plugins in self.json_info and plugin in self.json_info[_INFOJ.plugins]

    def get_plugin(self, plugin: str):
        if self.has_plugin(plugin):
            return self.json_info[_INFOJ.plugins][plugin]
        return None

    """params CRUD"""

    @property
    def params_fn(self):
        if self.has_plugin(_BUILTIN_PLUGIN.params):
            return os.path.join(self.root, _FNAME.params)

        return None

    @property
    @lru_cache()
    def params(self):
        from ..frame.params import BaseParams
        if self.has_plugin(_BUILTIN_PLUGIN.params):
            return BaseParams().from_json(self.params_fn)
        return None

    """tensorboard CRUD"""

    @property
    def board_logdir(self):
        if not self.has_board():
            return None
        writer = self.get_plugin(_BUILTIN_PLUGIN.writer)
        return writer[_PLUGIN_KEY.WRITER.log_dir]

    @property
    @lru_cache()
    def board_reader(self) -> BoardReader:

        logdir = self.board_logdir
        if logdir is None:
            return None

        fs = os.listdir(logdir)
        fs = [i for i in fs if i.endswith('.bd') and i.startswith('events.out.tfevents')]
        if len(fs) == 0:
            return None
        if len(fs) > 1:
            import warnings
            warnings.warn('found multi tfevents file, return the first one')
        file = os.path.join(logdir, fs[0])

        return BoardReader(file)

    def tensorboard(self):
        import subprocess
        with subprocess.Popen(['tensorboard', '--logdir={}'.format(self.board_logdir)], env=dict(os.environ)) as p:
            try:
                return p.wait(timeout=None)
            except:  # Including KeyboardInterrupt, wait handled that.
                p.kill()
                raise

    def has_board(self):
        return self.has_plugin(_BUILTIN_PLUGIN.writer) and self.has_dir(_PLUGIN_DIRNAME.writer)

    """logger CRUD"""

    @property
    def log_fn(self):
        if self.has_plugin(_BUILTIN_PLUGIN.logger):
            return self.get_plugin(_BUILTIN_PLUGIN.logger)[_PLUGIN_KEY.LOGGER.fn]
        return None

    def has_log(self):
        plugin_info = self.get_plugin(_BUILTIN_PLUGIN.logger)
        if plugin_info is not None:
            return os.path.exists(plugin_info[_PLUGIN_KEY.LOGGER.fn])

    """saver CRUD"""

    @property
    def saver_dir_fn(self):
        if self.has_plugin(_BUILTIN_PLUGIN.saver):
            return self.get_plugin(_BUILTIN_PLUGIN.saver)[_PLUGIN_KEY.SAVER.ckpt_dir]
        return None

    def delete_keypoints(self):
        dir = self.saver_dir_fn
        if dir is None:
            return
        fn = [os.path.join(dir, i) for i in os.listdir(dir) if i.endswith('.ckpt') and i.startswith('keep')]
        for f in fn:
            os.remove(f)
            jfn = '{}.json'.format(f)
            if os.path.exists(jfn):
                os.remove(jfn)

    def delete_checkpoints(self):
        dir = self.saver_dir_fn
        if dir is None:
            return
        fn = [os.path.join(dir, i) for i in os.listdir(dir) if i.endswith('.ckpt') and not i.startswith('keep')]
        for f in fn:
            os.remove(f)
            jfn = '{}.json'.format(f)
            if os.path.exists(jfn):
                os.remove(jfn)

    def delete_modules(self):
        dir = self.saver_dir_fn
        if dir is None:
            return
        fn = [os.path.join(dir, i) for i in os.listdir(dir) if i.startswith('model') and i.endswith('pth')]
        for f in fn:
            os.remove(f)
            jfn = '{}.json'.format(f)
            if os.path.exists(jfn):
                os.remove(jfn)

    def delete_saver_dir(self):
        dir = self.saver_dir_fn
        if dir is None:
            return
        shutil.rmtree(dir)

    def get_tag(self, tag: str):
        if self.has_tag(tag):
            return self.json_info[_INFOJ.tags][tag]
        return None

    """states CRUD"""

    def _build_state_fn(self, state_name: str):
        return '..{}'.format(state_name)

    @property
    def isfav(self):
        return self.has_state(_TEST_BUILTIN_STATE.fav)

    @property
    def visible(self):
        return not self.has_state(_TEST_BUILTIN_STATE.hide)

    @property
    def states(self) -> Dict[str, bool]:
        fn = [i for i in os.listdir(self.root) if i.startswith('..')]
        res = {}
        for f in fn:
            with open(os.path.join(self.root, f), 'r') as r:
                res[f] = bool(int(r.readline().strip()))
        return res

    def has_state(self, state_name: str) -> bool:
        fn = os.path.join(self.root, self._build_state_fn(state_name))
        if os.path.exists(fn):
            with open(fn, 'r') as w:
                state = int(w.readline().strip())
        else:
            state = 0
        return state == 1

    def toggle_state(self, state_name: str, toggle=None) -> bool:
        fn = os.path.join(self.root, self._build_state_fn(state_name))
        if os.path.exists(fn):
            with open(fn, 'r') as r:
                state = int(r.readline().strip())
            if toggle is None:
                new_state = 1 - state
            else:
                new_state = int(toggle)
        else:
            new_state = 1

        with open(fn, 'w') as w:
            w.write('{}'.format(new_state))

        return bool(new_state)

    def mark(self, state_name: str):
        self.toggle_state(state_name, True)

    def unmark(self, state_name: str):
        self.toggle_state(state_name, False)

    def hide(self):
        self.toggle_state(_TEST_BUILTIN_STATE.hide, True)

    def show(self):
        self.toggle_state(_TEST_BUILTIN_STATE.hide, False)

    def fav(self):
        self.toggle_state(_TEST_BUILTIN_STATE.fav, True)

    def unfav(self):
        self.toggle_state(_TEST_BUILTIN_STATE.fav, False)

    """git operation"""

    def reset(self) -> Experiment:
        """
        将工作目录中的文件恢复到某个commit
        恢复快照的 git 流程:
            git branch experiment
            git add . & git commit -m ... // 保证文件最新，防止冲突报错，这一步由 Experiment() 代为完成
            git checkout <commit-id> // 恢复文件到 <commit-id>
            git checkout -b reset // 将当前状态附到新的临时分支 reset 上
            git branch experiment // 切换回 experiment 分支
            git add . & git commit -m ... // 将当前状态重新提交到最新
                // 此时experiment 中最新的commit 为恢复的<commit-id>
            git branch -D reset  // 删除临时分支
            git branch master // 最终回到原来分支，保证除文件变动外git状态完好
        Returns:
            An Experiment represents this reset operation
        """
        commit = self.commit

        old_path = os.getcwd()
        os.chdir(commit.tree.abspath)
        exp = Experiment('Reset')

        repo = self.repo
        from thexp.utils.repository import branch
        with branch(commit.repo, _GITKEY.thexp_branch) as new_branch:
            repo.git.checkout(commit.hexsha)
            repo.git.checkout('-b', 'reset')
            repo.head.reference = new_branch
            repo.git.add('.')
            ncommit = repo.index.commit("Reset from {}".format(commit.hexsha))
            repo.git.branch('-d', 'reset')
        exp.add_plugin('reset', {
            'test_name': self.name,  # 从哪个状态恢复
            'from': exp.commit.hexsha,  # reset 运行时的快照
            'where': commit.hexsha,  # 恢复到哪一次 commit，是恢复前的保存的状态
            'to': ncommit.hexsha,  # 对恢复后的状态再次进行提交，此时 from 和 to 两次提交状态应该完全相同
        })

        exp.end()
        os.chdir(old_path)
        return exp

    def archive(self) -> Experiment:
        """
        将某次 test 对应 commit 的文件打包，相关命令为
            git archive -o <filename> <commit-hash>

        Returns:
            An Experiment represents this archive operation
        """
        commit = self.commit

        old_path = os.getcwd()
        os.chdir(commit.tree.abspath)
        exp = Experiment('Archive')

        revert_path = exp.makedir('archive')
        revert_fn = os.path.join(revert_path, "code.zip")
        exp.add_plugin('archive', {'file': revert_fn,
                                   'test_name': self.name})
        with open(revert_fn, 'wb') as w:
            self.repo.archive(w, commit)

        exp.end()
        os.chdir(old_path)
        return exp

    def tree(self, _=None):
        return self.json_info

    """for representation"""

    def df_columns(self):
        return [
            "success_exit",
            "start_time",
            "end_time",
            "argv",
            "project",
            "repo",
            "exp_name",
            "test_dir",
            "plugins",
            "short_hash",
        ]

    def df_info(self):
        res = list()
        json_info = self.json_info
        res.append(self.success_exit)
        res.append(json_info.get("start_time", None))
        res.append(json_info.get("end_time", None))
        argv = list(json_info.get("argv"))
        argv[0] = os.path.relpath(argv[0], json_info.get("repo"))

        res.append(' '.join(argv))
        res.append(json_info.get("project_name", None))
        res.append(json_info.get("repo"))
        res.append(json_info.get("exp_name", None))
        res.append(json_info.get("test_dir", None))
        res.append(', '.join(list(self.plugins.keys())))

        res.append(json_info.get("short_hash", None))

        return res
