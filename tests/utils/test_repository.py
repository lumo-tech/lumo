import tempfile
import os
import git

from lumo.proc.config import debug_mode
from lumo.utils.random import hashseed, int_time
from lumo.utils import repository
import random


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

    repository.git_commit(repo)
    # untracked_files
    a_str = write('a.txt')
    a_hash = repository.git_commit(repo)
    # uncommited chages
    b_str = write('a.txt')
    # untracked_files
    bb_str = write('b.txt')
    b_hash = repository.git_commit(repo)

    c_str = write('a.txt')
    # commited changes
    # load from current working directory
    c_hash = repository.git_commit(branch_name='main')
    d_hash = repository.git_commit(repo, branch_name='main')
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

    exp = repository.git_archive(repo, b_hash)
    zfile = exp.load_string('archive_fn')
    file = tarfile.open(zfile, mode='r')
    assert file.extractfile('a.txt').read().decode() == b_str
    assert file.extractfile('init.txt').read().decode() == f_str

    os.chdir(old_root)
