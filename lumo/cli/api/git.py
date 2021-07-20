"""
Methods about git.
"""
import git
import os
from functools import lru_cache
from typing import Tuple
from git import Repo, Commit

# from lumo.utils.keys import FN, CFG
from lumo.utils.paths import compare_path
from lumo.proc.path import git_dir

bin_file = ['*.pth', '*.npy', '*.ckpt',
            '*.ft',  # for feather
            '*.pkl'
            ]

lib_gitignores = ['.lumo/',
                  '*.lumo.*',
                  '.data',
                  '.datas',
                  '.dataset',
                  '.datasets',
                  ] + bin_file

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
                          '# Pyre type checker', '.pyre/'] + lib_gitignores)


def _check_gitignore(repo: Repo, force=False):
    """
    check if file `.gitignore`  have the needed ignored items for lumo.
    """
    # version_mark = os.path.join(repo.working_dir, FN.VERSION)
    # ignorefn = os.path.join(repo.working_dir, '.gitignore')

    # if os.path.exists(version_mark) and not force:
    #     if os.path.exists(ignorefn):
    #         return False

    old_marks = [f for f in os.listdir(repo.working_dir) if f.startswith('.lumo.')]
    for old_mark in old_marks:
        mark_fn = os.path.join(repo.working_dir, old_mark)
        if os.path.isfile(mark_fn):
            os.remove(mark_fn)

    with open(version_mark, 'w') as w:
        pass

    if not os.path.exists(ignorefn):
        with open(ignorefn, 'w', encoding='utf-8') as w:
            w.write(py_gitignore)
        return True
    else:
        amend = False
        with open(ignorefn, 'r', encoding='utf-8') as r:
            lines = [i.strip() for i in r.readlines()]
            for item in lib_gitignores:
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


def init_repo(dir='./') -> Tuple[Repo, bool]:
    """
    initialize a directory, including git init, lumo config and a initial commit.
    """
    path = git_dir(dir)
    init = False
    if path is not None and compare_path(path, dir):
        repo = Repo(path)
    else:
        repo = Repo.init(dir)
        init = True

    if _check_gitignore(repo=repo, force=True):
        repo.git.add('.')
        repo.index.commit('initial commit')
        init = True

    check_have_commit(repo)
    return repo, init
