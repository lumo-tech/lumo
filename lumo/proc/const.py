import os
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

    IS_PYCHARM = os.environ.get("PYCHARM_HOSTED", '0') == "1"
    IS_PYCHARM_DEBUG = eval(os.environ.get('IPYTHONENABLE', "False"))

    ENV = "jupyter" if "jupyter_core" in sys.modules else "python"

    IS_GIT_ENABLED = (os.environ.get('LUMO_GIT', '1') == '1' and
                      os.environ.get('LUMO_NOGIT', '0') == '0')


class CFG:
    class PATH:  # key for path
        GLOBAL_EXP = 'global_exp'
        DATASET = 'datasets'
        PRETRAINS = 'pretrains'
        CACHE = 'cache'

    class FIELD:
        RUNTIME = 'runtime'
        GLOBAL = 'global'
        DEFAULT = 'default'

    BRANCH_NAME = 'experiment'


class FN:
    PHASH = '.hash'
    # CONFIGJS = 'config.json'
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


class EXP_CONST:
    class INFO_KEY:
        STATE = 'state'  # dumped info key
        PROJECT = 'project'
        EXECUTE = 'execute'
        GIT = 'git'
        VERSION = 'version'
        TRAINER = 'trainer'

    class IO_DIR:
        SINFO_DIR = '.json'
        PKL_DIR = '.pkl'
        INFO_DIR = '.line'

    # class HOOK_FN