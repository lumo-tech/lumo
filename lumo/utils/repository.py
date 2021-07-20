"""
Methods about git.
"""
import git
import os
from functools import lru_cache
from lumo.base_classes import attr

try:
    from git import Repo, Commit
except ImportError:
    def Repo(*_, **__):
        return None


    def Commit(*_, **__):
        return None

from lumo.proc.explore import git_enable
from lumo.proc.path import git_dir, cache_dir
from lumo.utils.paths import checkpath


class GitKey:
    LUMO_BRANCH = 'lumo_experiments'


class GitCMDResult(attr):
    def __init__(self, cmd, result=0, msg=''):
        self.cmd = cmd
        self.result = result
        self.msg = msg

    @classmethod
    def failed(cls, cmd, msg):
        return cls(cmd, 1, msg)


class branch:
    """
    用于配合上下文管理切换 git branch

    with branch(repo, branch):
        repo.index.commit('...')
    """

    def __init__(self, repo, branch: str):
        self.repo = repo
        self.old_branch = self.repo.head.reference
        self.branch = branch

    def __enter__(self):
        if self.branch not in self.repo.heads:
            head = self.repo.create_head(self.branch)
        else:
            head = self.repo.heads[self.branch]

        self.repo.head.reference = head
        # self.repo.head.reset(index=True, working_tree=True)
        return head

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.repo.head.reference = self.old_branch


def check_have_commit(repo):
    if len(repo.heads) == 0:
        repo.git.add('.')
        repo.index.commit('initial commit')


@lru_cache()
def load_repo(dir='./'):
    """
    Try to load git repository object of a directory.
    Args:
        dir: str, a directory path, default is the current working dir.
        if dir is a repository dir, then a git.Repo object will be retured.
        if not, some you can type a path to init it, or type '!' to cancel init it.

    Returns:
        git.Repo object or None if dir not have git repository and cancel to init it.
    """
    path = git_dir(dir)
    repo = Repo(path)
    check_have_commit(repo)
    return repo


class GitWrap:
    def add(self, repo=None):
        raise NotImplementedError()

    def commit(self, repo=None, key=None, branch_name='lumo_experiments', info: str = None):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def archive(self):
        raise NotImplementedError()


class GitDisabledWrap(GitWrap):
    def add(self, repo=None):
        return GitCMDResult('add', 1, 'git cannot be used.')

    def commit(self, repo=None, key=None, branch_name='lumo_experiments', info: str = None):
        return GitCMDResult('commit', 1, 'git cannot be used.')

    def reset(self):
        return GitCMDResult('reset', 1, 'git cannot be used.')

    def archive(self):
        return GitCMDResult('archive', 1, 'git cannot be used.')


class GitEnabledWrap(GitWrap):

    def add(self, repo=None):
        if repo is None:
            repo = load_repo()
        return repo.git.add(all=True)

    def commit(self, repo=None, key=None, branch_name=GitKey.LUMO_BRANCH, info: str = None):
        """
        ```
            TODO behavior need to be verified.
            cd <repo working dir>
            git reset <branch_name>
            git add .
            git commit -m "<info>"
            git reset <original branch>
        ```
        Args:
            repo:
            key:
                to avoid duplicate commit, a key value can be passed,
                commit operation will be perform if `key` hasn't appeared before.
                default value is None, means you can commit as you want without limitation
            branch_name:
                commit on which branch, nonexistent branch will be created.
            info:
                commit info, a string
        Returns:
            git.Commit object, see gitpython for details.
        """
        try:
            if repo is None:
                repo = load_repo()

            if key is not None and key in _commits_map:
                return _commits_map[key]

            with branch(repo, branch_name):
                repo.git.add(all=True)
                commit_info = '[[EMPTY]]'
                if info is not None:
                    commit_info = info
                commit_ = repo.index.commit(commit_info)
            if key is not None:
                _commits_map[key] = commit_
        except git.GitCommandError:
            commit_ = None
        return commit_

    def reset(self, repo=None, commit_hex=None, commit=None):
        """
        TODO
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
        commit = commit
        if repo is None:
            repo = load_repo()
        old_path = os.getcwd()
        os.chdir(commit.tree.abspath)
        # exp = Experiment('Reset')

        from thexp.utils.repository import branch
        with branch(commit.repo, GitKey.LUMO_BRANCH) as new_branch:
            repo.git.checkout(commit.hexsha)
            repo.git.checkout('-b', 'reset')
            repo.head.reference = new_branch
            repo.git.add('.')
            ncommit = repo.index.commit("Reset from {}".format(commit.hexsha))
            repo.git.branch('-d', 'reset')
        # exp.add_plugin('reset', {
        #     'test_name': self.name,  # 从哪个状态恢复
        #     'from': exp.commit.hexsha,  # reset 运行时的快照
        #     'where': commit.hexsha,  # 恢复到哪一次 commit，是恢复前的保存的状态
        #     'to': ncommit.hexsha,  # 对恢复后的状态再次进行提交，此时 from 和 to 两次提交状态应该完全相同
        # })
        # exp.end()

        os.chdir(old_path)
        return None

    def archive(self, repo=None, commit_hex=None, commit=None, tgt=None):
        """
        TODO
        将某次 test 对应 commit 的文件打包，相关命令为
            git archive -o <filename> <commit-hash>
        Returns:
            An Experiment represents this archive operation
        """
        if repo is None:
            repo = load_repo()

        old_path = os.getcwd()
        os.chdir(commit.tree.abspath)
        # exp = Experiment('Archive')

        revert_path = checkpath(cache_dir(), 'archives', commit)
        revert_fn = os.path.join(revert_path, "code.zip")

        # TODO 在code.zip目录下添加相关说明
        # exp.add_plugin('archive', {'file': revert_fn,
        #                            'test_name': self.name})
        with open(revert_fn, 'wb') as w:
            repo.archive(w, commit)

        # exp.end()
        os.chdir(old_path)
        return None


if git_enable():
    wrap = GitEnabledWrap()
else:
    wrap = GitDisabledWrap()

add = wrap.add
commit = wrap.commit
reset = wrap.reset
archive = wrap.archive

_commits_map = {}
