"""
Methods about git.
"""
import json
import os
import sys
from functools import lru_cache
from typing import List
from uuid import uuid4

from git import Git, Repo
from thexp import __VERSION__
from thexp.utils.paths import renormpath
from .dates import curent_date
from ..globals import _GITKEY, _OS_ENV, _FNAME

torch_file = ['*.pth', '*.npy', '*.ckpt']
thexp_gitignores = ['.thexp/', _FNAME.repo, _FNAME.expsdirs, '.idea/', '*.thexp.*', '*.pkl'] + torch_file

py_gitignore = "\n".join(['# Byte-compiled / optimized / DLL files',
                          '__pycache__/', '*.py[cod]',
                          '*$py.class',
                          '# C extensions', '*.so',
                          '# Distribution / packaging',
                          '.Python', 'build/', 'develop-eggs/', 'dist/', 'downloads/', 'eggs/', '.eggs/',
                          'lib/', 'lib64/', 'parts/', 'sdist/', 'var/', 'wheels/', 'pip-wheel-metadata/',
                          'share/python-wheels/', '*.egg-info/', '.installed.cfg', '*.egg', 'MANIFEST',
                          '', '# PyInstaller',
                          '#  Usually these files are written by a python script from a template',
                          '#  before PyInstaller builds the exe, so as to inject date/other infos into '
                          'it.',
                          '*.manifest', '*.spec',
                          '# Installer logs', 'pip-log.txt',
                          'pip-delete-this-directory.txt',
                          '# Unit test / coverage reports',
                          'htmlcov/', '.tox/', '.nox/', '.coverage', '.coverage.*', '.cache',
                          'nosetests.xml', 'coverage.xml', '*.cover', '.hypothesis/', '.pytest_cache/',
                          '# Translations', '*.mo', '*.pot',
                          '# Django stuff:', '*.log', 'local_settings.py', 'db.sqlite3',
                          '# Flask stuff:',
                          'instance/', '.webassets-cache', '', '# Scrapy stuff:', '.scrapy', '',
                          '# Sphinx documentation', 'docs/_build/', '', '# PyBuilder', 'target/', '',
                          '# Jupyter Notebook', '.ipynb_checkpoints', '',
                          '# IPython',
                          'profile_default/', 'ipython_config.py',
                          '# pyenv', '.python-version', '',
                          '# celery beat schedule file', 'celerybeat-schedule', '',
                          '# SageMath parsed files', '*.sage.py', '',
                          '# Environments', '.env', '.venv',
                          'env/', 'venv/', 'ENV/', 'env.bak/', 'venv.bak/', '',
                          '# Spyder project settings', '.spyderproject', '.spyproject', '',
                          '# Rope project settings', '.ropeproject', '',
                          '# mkdocs documentation',
                          '/site', '',
                          '# mypy',
                          '.mypy_cache/', '.dmypy.json', 'dmypy.json',
                          '# Pyre type checker', '.pyre/'] + thexp_gitignores)


def _set_default_config(repo: Repo, names: List[str]):
    """
    set default git config, including:
    1. [expsdir], represent the root path of the experiment extra data(saved by thexp.frame.Experiment)
    of this repository

    2. [uuid], to avoid duplicate name, all repository will have another 2-bit hash value produced by uuid.

    3. [projname], record the directory name of this repository to guarantee the key flag of this repository won't change.
    """
    if len(names) == 0:
        return {}

    def _expsdir(repo):
        from .paths import global_config
        config = global_config()
        expsdir = config.get(_GITKEY.expsdir, renormpath(os.path.join(repo.working_dir, '.thexp/experiments')))
        return expsdir

    _default = {
        _GITKEY.uuid: lambda *args: uuid4().hex[:2],
        _GITKEY.expsdir: lambda repo: _expsdir(repo),
        _GITKEY.projname: lambda repo: renormpath(os.path.basename(repo.working_dir))
    }

    writer = repo.config_writer()
    res = {}
    for name in names:
        value = _default[name](repo)
        writer.add_value(_GITKEY.section_name, name, value)
        res[name] = value

    writer.write()
    writer.release()

    return res


def _check_section(repo: Repo):
    """
    check if git config section([thexp]) exists.
    """
    writer = repo.config_writer()
    if not writer.has_section(_GITKEY.section_name):
        writer.add_section(_GITKEY.section_name)
    writer.write()
    writer.release()


@lru_cache()
def git_config(repo: Repo):
    """
    load the git config of the given repository.

    Because of functools.lru_cache, this time-cosuming function will only read git config once from disk
    for each repository. And it will then load from lru_cache.
    """
    reader = repo.config_reader()
    if not reader.has_section(_GITKEY.section_name):
        _check_section(repo)
        reader = repo.config_reader()
    reader.read()
    try:
        config = {k: v for k, v in reader.items(_GITKEY.section_name)}
    except:
        config = {}

    lack_names = [i for i in {_GITKEY.expsdir, _GITKEY.uuid, _GITKEY.projname} if i not in config]
    _updates = _set_default_config(repo, lack_names)

    config.update(_updates)

    return config


def check_gitignore(repo: Repo, force=False):
    """
    check if file `.gitignore`  have the needed ignored items for thexp.
    """
    rp = os.path.join(repo.working_dir, _FNAME.expsdirs)

    version_mark = os.path.join(repo.working_dir, _FNAME.gitignore_version)
    if os.path.exists(rp) and os.path.exists(version_mark) and not force:
        return

    old_marks = [f for f in os.listdir(repo.working_dir) if f.startswith('.thexp.')]
    for old_mark in old_marks:
        mark_fn = os.path.join(repo.working_dir, old_mark)
        if os.path.isfile(mark_fn):
            os.remove(mark_fn)

    with open(version_mark, 'w') as w:
        pass

    ignorefn = os.path.join(repo.working_dir, _FNAME.gitignore)
    if not os.path.exists(ignorefn):
        with open(ignorefn, 'w', encoding='utf-8') as w:
            w.write(py_gitignore)
        return True
    else:
        amend = False
        with open(ignorefn, 'r', encoding='utf-8') as r:
            lines = [i.strip() for i in r.readlines()]
            for item in thexp_gitignores:
                if item not in lines:
                    lines.append(item)
                    amend = True
        if amend:
            with open(ignorefn, 'w', encoding='utf-8') as w:
                w.write('\n'.join(lines))

    return amend


def git_config_syntax(value: str) -> str:
    """
    syntax value(especially file path) which will be stored in git config
    """
    return value.replace('\\\\', '/').replace('\\', '/')


@lru_cache()
def git_root(dir="./", ignore_info=False):
    """
    判断某目录是否在git repo 目录内（包括子目录），如果是，返回该 repo 的根目录
    :param dir:  要判断的目录。默认为程序运行目录
    :return: 如果是，返回该repo的根目录（包含 .git/ 的目录）
        否则，返回空
    """
    cur = os.getcwd()
    os.chdir(dir)
    try:
        res = Git().execute(['git', 'rev-parse', '--git-dir'])
    except Exception as e:
        if not ignore_info:
            print(e)
        res = None
    os.chdir(cur)
    return res


class branch:
    """
    用于配合上下文管理切换 git branch

    with branch(repo, branch):
        repo.index.commit('...')
    """

    def __init__(self, repo: Repo, branch: str):
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


def init_repo(dir='./'):
    """
    initialize a directory, including git init, thexp config and a initial commit.
    """
    path = git_root(dir, ignore_info=True)
    if path is not None:
        res = Repo(path)
    else:
        res = Repo.init(path)
    check_gitignore(repo=res, force=True)
    git_config(res)
    res.git.add('.')
    res.index.commit('initial commit')
    return res


@lru_cache()
def load_repo(dir='./') -> Repo:
    """
    Try to load git repository object of a directory.
    Args:
        dir: str, a directory path, default is the current working dir.
        if dir is a repository dir, then a git.Repo object will be retured.
        if not, some you can type a path to init it, or type '!' to cancel init it.

    Returns:
        git.Repo object or None if dir not have git repository and cancel to init it.
    """

    path = git_root(dir)

    if path is None:
        if _OS_ENV.IGNORE_REPO not in os.environ:
            print("fatal: not a git repository (or any of the parent directories)")
            print("-----------------------")
            path = input("type root path to init this project, \n(default: {}, type '!' to ignore".format(os.getcwd()))
        else:
            print("Variable 'repo' will be a None object. "
                  "Any operation that need this repo may cause Exception.")
            return None

        if '!' in path or '！' in path:
            print("Variable 'repo' will be a None object. "
                  "Any operation that need this repo may cause Exception.")
            return None

        res = Repo.init(path)
        check_gitignore(repo=res, force=True)
        # check_gitconfig(repo=res, force=True)
        res.git.add('.')
        res.index.commit('initial commit')
    else:
        res = Repo(path)
        amend = check_gitignore(repo=res, force=False)
        if amend:
            res.git.add(_FNAME.gitignore)
            res.index.commit('fix gitignore')
    return res


_commits_map = {}


def commit(repo: Repo, key=None, branch_name=_GITKEY.thexp_branch):
    """"""
    if key is not None and key in _commits_map:
        return _commits_map[key]

    with branch(repo, branch_name):
        repo.git.add(all=True)
        commit_date = curent_date()
        commit_info = dict(
            date=commit_date,
            args=sys.argv,
            environ="jupyter" if "jupyter_core" in sys.modules else "python",
            version=sys.version,
        )
        commit_ = repo.index.commit(json.dumps(commit_info, indent=2))
    if key is not None:
        _commits_map[key] = commit_
    return commit_
