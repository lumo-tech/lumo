import os
import torch
import numpy as np
import sys
from sys import platform
from typing import Mapping, TypeVar, Type

T = TypeVar('T')

from lumo import __version__
from lumo.contrib.itertools import lfilter

LIBRARY_NAME = 'lumo'


class ENV:
    IS_WIN = (platform == "win32")
    IS_MAC = (platform == "darwin")
    IS_LINUX = (platform == "linux" or platform == "linux2")

    IS_REMOTE = any([i in os.environ for i in ['SHELL',
                                               'SHLVL',
                                               'SSH_CLIENT',
                                               'SSH_CONNECTION',
                                               'SSH_TTY']])
    IS_LOCAL = not IS_REMOTE

    IS_PYCHARM = os.environ.get("PYCHARM_HOSTED", 0) == "1"
    IS_PYCHARM_DEBUG = eval(os.environ.get('IPYTHONENABLE', "False"))

    ENV = "jupyter" if "jupyter_core" in sys.modules else "python"

    HAS_GIT = True  # TODO check state


def join(*args):
    args = [i for i in args if i is not None]
    return "_".join(args)


class CFG:
    class PATH:  # key for path
        GLOBAL_EXP = 'global_exp'
        LOCAL_EXP = 'local_exp'
        EXP_ROOT = 'exp_root'  # local first, then global
        REPO = 'repo'  # project_root
        CWD = 'working_dir'
        DATASET = 'datasets'
        PRETRAINS = 'pretrains'
        TMPDIR = 'TMPDIR'
        TMP_DIR = 'TMP_DIR'
        CACHE = 'cache'

        class DEFAULT:  # default values
            GLOBAL_EXP = os.path.expanduser("~/.lumo/experiments")
            DATASET = os.path.expanduser("~/.lumo/datasets")
            PRETRAINS = os.path.expanduser("~/.lumo/pretrains")
            CACHE = os.path.expanduser("~/.lumo/.cache")
            LOCAL_CACHE = '.cache'

    class STATE:
        DISABLE_GIT = 'nocommit'

        class OS_NAME:
            DISABLE_GIT = 'LUMO_NOCOMMIT'

        class DEFAULT:
            DISABLE_GIT = not ENV.HAS_GIT

    class FIELD:
        RUNTIME = 'runtime'
        REPO = 'local'
        GLOBAL = 'global'

    REPO_NAME = 'name'
    BRANCH_NAME = 'experiment'


class FN:
    PHASH = '.hash'
    CONFIGJS = 'config.json'
    REPOSJS = 'repos.json'
    TESTLOG = 'tests.log'
    VERSION = f'.lumo.{__version__}'

    D_LINE = '.line'
    D_JSON = '.json'
    D_PKL = '.pkl'

    class SUFFIX:
        D_LINE = '.txt'
        D_JSON = '.json'
        D_PKL = '.pkl'


class EXP:  # used in Experiment
    STATE = 'state'  # dumped info key
    EXCEPTION = 'exception'
    PROJECT = 'project'
    EXECUTE = 'execute'
    GIT = 'git'

    VERSION = 'version'
    UUID = 'uuid'

    TRAINER = 'trainer'


class DIST:
    RANK = 'local_rank'


class DL:
    XS = 'xs'
    INPUTS = 'inputs'
    OUTPUTS = 'outputs'
    LOGITS = 'logits'
    FEAT = 'features'
    MODEL = 'model'

    TOKEN_IDS = 'token_ids'


class RESERVE:
    ATTR_TYPE = '_type'


class TRAINER:
    path = 'path'
    doc = 'doc'
    basename = 'basename'
    class_name = 'class_name'

    class DKEY:
        models = 'models'
        optims = 'optims'
        tensor = 'tensor'
        others = 'others'
        initial = 'initial'
        train_epoch_toggle = 'train_epoch_toggle'
        train_toggle = 'train_toggle'
        device = 'device'
        devices = 'devices'
        ini_models = 'initial.models'
        ini_datasets = 'initial.datasets'
        ini_callbacks = 'initial.callbacks'
        params = 'params'
        exp = 'exp'


class K:
    @staticmethod
    def join(*args):
        return '_'.join([i for i in args if i is not None])

    @staticmethod
    def get_str(mem: Mapping, key, default=None) -> str:
        return mem.get(key, default)

    @staticmethod
    def get_pt(mem: Mapping, key, default=None) -> torch.Tensor:
        return mem.get(key, default)

    @staticmethod
    def get_np(mem: Mapping, key, default=None) -> np.ndarray:
        return mem.get(key, default)

    @staticmethod
    def get_type(mem: Mapping, key, default=None, *, type: Type[T]) -> T:
        return mem.get(key, default)
