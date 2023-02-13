"""
Methods about git.
"""
import os
import warnings
from functools import lru_cache

import git
from git import Repo, Commit
import io
from joblib import hash
from .filelock2 import Lock

LUMO_BRANCH = 'lumo_experiments'

_commits_map = {}


class branch:
    """
    用于配合上下文管理切换 git branch

    with branch(repo, branch):
        repo.index.commit('...')
    """

    def __init__(self, repo: Repo, branch: str):
        self.repo = repo
        self.lock = Lock(f'{hash(repo.git_dir)}_{branch}')
        self.lock.abtain()
        self.old_branch = self.repo.head.reference
        self.branch = branch

    def __enter__(self):
        if self.branch is None:
            return

        if self.branch not in self.repo.heads:
            head = self.repo.create_head(self.branch)
        else:
            head = self.repo.heads[self.branch]

        self.repo.head.set_reference(head)
        # self.repo.head.reset(index=True, working_tree=True)
        return head

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.branch is None:
            return
        self.repo.head.set_reference(self.old_branch)
        self.lock.release()


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


def add(repo=None):
    if repo is None:
        repo = load_repo()
    return repo.git.add(all=True)


def git_commit(repo=None, key=None, branch_name=LUMO_BRANCH, info: str = None, filter_files=None):
    """
    ```
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

        if LUMO_BRANCH not in repo.branches:
            repo.create_head(LUMO_BRANCH)
            print(f'branch {LUMO_BRANCH} not found, will be created automatically.')

        exp_head_commit = repo.heads[LUMO_BRANCH].commit
        diff = repo.active_branch.commit.diff(exp_head_commit)

        if filter_files is not None:
            diff = [i.a_path for i in diff if i.a_path in filter_files]

        if len(diff) == 0:
            commit_ = exp_head_commit
        else:
            with branch(repo, branch_name):
                change_file = []
                change_file.extend(repo.untracked_files)
                change_file.extend([i.a_path for i in repo.head.commit.diff(None)])
                # print(change_file)
                if filter_files is not None:
                    print('before filter', change_file)
                    change_file = [i for i in change_file if i in filter_files]
                print('after filter', change_file)

                repo.git.add(change_file)
                commit_info = '[[EMPTY]]'
                if info is not None:
                    commit_info = info
                commit_ = repo.index.commit(commit_info)

        if key is not None:
            _commits_map[key] = commit_
    except (git.GitCommandError, ValueError, IndexError) as e:
        commit_ = None
        print(f'git commit Exception {e}')
    return commit_


def reset(repo=None, commit_hex=None, commit: Commit = None):
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
    if repo is None:
        repo = load_repo()

    if commit is None and commit_hex is not None:
        commit = repo.commit(commit_hex)

    old_path = os.getcwd()
    os.chdir(commit.tree.abspath)

    with branch(commit.repo, LUMO_BRANCH) as new_branch:
        repo.git.checkout(commit.hexsha)
        repo.git.checkout('-b', 'reset')
        repo.head.reference = new_branch
        _ = git_commit(repo, branch_name=repo.head.reference.name, info="Reset from {}".format(commit.hexsha))
        repo.git.branch('-d', 'reset')

    os.chdir(old_path)
    return None


def archive(repo=None, commit_hex=None, commit: Commit = None, tgt=None):
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

    # revert_path = checkpath(cache_dir(), 'archives', commit)
    # revert_fn = os.path.join(revert_path, "code.zip")

    # TODO 在code.zip目录下添加相关说明
    # exp.add_plugin('archive', {'file': revert_fn,
    #                            'test_name': self.name})
    # with open(revert_fn, 'wb') as w:
    #     repo.archive(w, commit)

    # exp.end()
    os.chdir(old_path)
    return None


@lru_cache(1)
def git_enable():
    try:
        import git
    except ImportError:
        warnings.warn('python library `gitpython` not installed, git operations will be ignored. '
                      'If you want lumo to use git to manage your code, please install it by executing `pip install gitpython`')
        return False

    try:
        git.Git().execute(['git', 'rev-parse', '--git-dir'])
        return True
    except git.GitCommandError:
        return False


def git_dir(root='./'):
    """
    git repository directory
    git rev-parse --git-dir
    Args:
        root:
    Returns:

    """
    if git_enable():
        from git import Git
        cur = os.getcwd()
        os.chdir(root)
        res = Git().execute(['git', 'rev-parse', '--git-dir'])
        res = os.path.abspath(os.path.dirname(res))
        os.chdir(cur)
        return res
    else:
        return None


def get_tree_from_commit(commit: Commit, tree=None):
    if tree is None:
        tree = commit.tree
    yield tree.abspath, tree.blobs, tree.trees
    for tree in tree.trees:
        yield from get_tree_from_commit(commit, tree)


def get_diff_tree_from_commits():
    pass


def get_file_of_commit(commit: Commit, file_name) -> bytes:
    blob = commit.tree / file_name
    return blob.data_stream.read()
