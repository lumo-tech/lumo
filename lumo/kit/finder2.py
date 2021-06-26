from datetime import datetime
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

from git import Commit, Repo
from lumo.base_classes import attr
from lumo.utils import safe_io as io
from lumo.utils.dates import date_from_str
from lumo.utils.keys import FN, EXP
from lumo.utils.paths import home_dir, compare_path
from lumo.utils.repository import load_repo, repo_dir


class TestProp():
    @property
    def command(self):
        raise NotImplementedError()

    @property
    def exp_name(self):
        raise NotImplementedError()

    @property
    def exp_root(self):
        raise NotImplementedError()

    @property
    def project_name(self):
        raise NotImplementedError()

    @property
    def repo_root(self):
        raise NotImplementedError()

    @property
    def repo_hash(self):
        raise NotImplementedError()

    @property
    def start_time(self):
        raise NotImplementedError()

    @property
    def end_time(self):
        raise NotImplementedError()

    @property
    def end_code(self):
        raise NotImplementedError()

    @property
    def grepo(self) -> Repo:
        raise NotImplementedError()

    @property
    def gcommit(self) -> Commit:
        raise NotImplementedError()

    @property
    def commit_hash(self) -> str:
        raise NotImplementedError()

    @property
    @lru_cache(1)
    def lines(self) -> Dict[str, Dict]:
        raise NotImplementedError()

    @property
    @lru_cache(1)
    def jsons(self):
        raise NotImplementedError()

    @property
    def pkl_keys(self):
        raise NotImplementedError()

    @property
    def uuid(self):
        raise NotImplementedError()

    @property
    def largest_epoch(self):
        raise NotImplementedError()
