import os
import random
import tempfile

import git

from lumo.proc.config import debug_mode
from lumo.utils import repository


def write(fn):
    with open(fn, 'w') as w:
        st = str(random.random())
        w.write(st)
    return st


def read(fn):
    with open(fn) as w:
        return w.read()


def test_git():
    debug_mode()
    git_root = tempfile.mkdtemp()
    old_root = os.getcwd()
    os.chdir(git_root)
    repo = git.Repo.init(git_root)
    f_str = write('init.txt')
    repo.index.add(['init.txt'])
    repo.index.commit('initial commit')
    main_branch = repo.active_branch.name

    repository.git_commit(repo)
    # untracked_files
    a_str = write('a.txt')
    a_hash = repository.git_commit(repo)
    # uncommitted changes
    b_str = write('a.txt')
    # untracked_files
    bb_str = write('b.txt')
    b_hash = repository.git_commit(repo)

    c_str = write('a.txt')
    # committed changes
    c_hash = repository.git_commit(repo, branch_name=main_branch)
    d_hash = repository.git_commit(repo, branch_name=main_branch)
    cc_hash = repository.git_commit(repo)
    assert c_hash == d_hash
    old_branch_name = repository.git_checkout(repo, a_hash)

    assert repo.active_branch.name == old_branch_name
    assert read('a.txt') == a_str
    old_branch_name = repository.git_checkout(repo, b_hash)
    assert repo.active_branch.name == old_branch_name
    assert read('a.txt') == b_str
    assert read('b.txt') == bb_str

    old_branch_name = repository.git_checkout(repo, cc_hash)
    assert repo.active_branch.name == old_branch_name
    assert read('a.txt') == c_str
    assert read('b.txt') == bb_str

    import tarfile

    archived_fn = repository.git_archive(tempfile.mkdtemp(), repo, b_hash)
    file = tarfile.open(archived_fn, mode='r')
    assert file.extractfile('a.txt').read().decode() == b_str
    assert file.extractfile('init.txt').read().decode() == f_str

    os.chdir(old_root)
