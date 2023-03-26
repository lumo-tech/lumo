"""
Methods about git.
"""
import os
import warnings
from functools import lru_cache

import git
from git import Repo, Commit
from joblib import hash

from .filelock import Lock


def dev_branch():
    """
    Returns the value of the 'dev_branch' key from the global configuration dictionary 'glob' in the 'lumo.proc.config' module.

    If the key is not present in the dictionary, returns the default value of 'lumo_experiments'.

    Returns:
        str: The value of the 'dev_branch' key from the global configuration dictionary. By default is `lumo_experiments`.

    """
    from lumo.proc.config import glob
    return glob.get('dev_branch', 'lumo_experiments')


_commits_map = {}


class branch:
    """
    A context manager class for switching git branches in a given repository.

    Example usage:
        with branch(repo, branch):
            repo.index.commit('...')

    Args:
        repo (Repo): The repository object for which the branch will be switched.
        branch (str): The name of the branch to switch to.


    Notes:
        This class provides a context manager that switches the current branch
         to the given branch when entering the context and switches back to the original branch when exiting the context.

        If the given branch does not exist in the repository, it will be created.

        A lock is obtained on the repository to ensure that
        only one instance of this class can switch branches at a time for a given repository.

    """

    def __init__(self, repo: Repo, branch: str):
        self.repo = repo
        self.lock = Lock(f'{hash(repo.git_dir)}_lumo_auto_commit')
        self.lock.abtain()
        self.old_branch = self.repo.head.reference
        self.branch = branch

    def __enter__(self):
        if self.branch == self.old_branch.name:
            return
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
        if self.branch == self.old_branch.name:
            return

        if self.branch is None:
            return
        self.repo.head.set_reference(self.old_branch)
        self.lock.release()


def check_have_commit(repo):
    """
    Checks if the given repository has any commits.

    If there are no commits, creates an initial commit that adds all files in the repository and has the message "initial commit".
    """
    if len(repo.heads) == 0:
        repo.git.add('.')
        repo.index.commit('initial commit')


def load_repo(root='./'):
    """
    Try to load git repository object of a directory.
    Args:
        root: str, a directory path, default is the current working dir.
        if dir is a repository dir, then a git.Repo object will be retured.
        if not, some you can type a path to init it, or type '!' to cancel init it.

    Returns:
        git.Repo object or None if dir not have git repository and cancel to init it.
    """
    path = git_dir(root)
    repo = Repo(path)
    check_have_commit(repo)
    return repo


def git_commit(repo=None, key=None, branch_name=None, info: str = None, filter_files=None):
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
    if branch_name is None:
        branch_name = dev_branch()

    try:
        if repo is None:
            repo = load_repo()

        if key is not None and key in _commits_map:
            return _commits_map[key]

        if branch_name not in repo.branches:
            repo.create_head(branch_name)
            print(f'branch {branch_name} not found, will be created automatically.')

        diff_uncommit = repo.head.commit.diff()
        exp_head_commit = repo.heads[branch_name].commit
        diff_from_branches = repo.active_branch.commit.diff(exp_head_commit)
        # print(diff_uncommit)

        if filter_files is not None:
            diff_from_branches = [i for i in diff_from_branches if i.a_path in filter_files]

        if len(diff_from_branches) == 0 and len(diff_uncommit) == 0 and len(repo.untracked_files) == 0:
            commit_ = exp_head_commit
        else:
            with branch(repo, branch_name):
                change_file = []
                change_file.extend(repo.untracked_files)
                change_file.extend([i.a_path for i in diff_from_branches])
                change_file.extend([i.a_path for i in diff_uncommit])
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


def git_archive(target_path, repo=None, commit_hex=None, commit: Commit = None):
    """
    git archive -o <filename> <commit-hash>

    Returns:
        An Experiment represents this archive operation
    """
    if repo is None:
        repo = load_repo()

    if commit is None and commit_hex is not None:
        commit = repo.commit(commit_hex)

    old_path = os.getcwd()
    os.chdir(commit.tree.abspath)

    fn = os.path.join(target_path, f'{commit.hexsha[:8]}.tar')

    with open(fn, 'wb') as w:
        repo.archive(w, commit.hexsha)

    os.chdir(old_path)
    return fn


def git_checkout(repo=None, commit_hex=None, commit: Commit = None):
    """
    Checkout a specific commit in a Git repository.

    Args:
        repo (git.Repo, optional): The Git repository to use. Defaults to None, in which case the repository is loaded using `load_repo()`.
        commit_hex (str, optional): The hash of the commit to check out. Defaults to None.
        commit (git.Commit, optional): The commit object to check out. Defaults to None.

    Returns:
        str: The abbreviated hash of the checked-out commit.

    Raises:
        git.InvalidGitRepositoryError: If the specified repository is invalid or not found.
        git.BadName: If the specified branch name is invalid or not found.
    """
    if repo is None:
        repo = load_repo()

    if commit is None and commit_hex is not None:
        commit = repo.commit(commit_hex)

    old_path = os.getcwd()
    os.chdir(commit.tree.abspath)

    # with branch(commit.repo, LUMO_BRANCH) as new_branch:
    repo.git.checkout('-b', commit.hexsha[:8], commit.hexsha)

    os.chdir(old_path)
    return commit.hexsha[:8]


@lru_cache(1)
def git_enable():
    """
    Check if Git is installed and a repository is present.

    Returns:
        bool: True if Git is installed and a repository is present, False otherwise.

    Raises:
        ImportError: If the `gitpython` library is not installed.
    """
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
    git rev-parse --show-toplevel
    Args:
        root:
    Returns:

    The original command, `git rev-parse --git-dir`, can not find a right path when the repository is a submodule inside another repository.
    """
    if git_enable():
        from git import Git
        cur = os.getcwd()
        os.chdir(root)
        res = Git().execute(['git', 'rev-parse', '--show-toplevel'])
        res = os.path.abspath(res)
        os.chdir(cur)
        return res
    else:
        return None

# def get_tree_from_commit(commit: Commit, tree=None):
#     if tree is None:
#         tree = commit.tree
#     yield tree.abspath, tree.blobs, tree.trees
#     for tree in tree.trees:
#         yield from get_tree_from_commit(commit, tree)
