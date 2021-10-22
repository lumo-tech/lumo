import os
import sys
from sys import platform
from typing import TypeVar

T = TypeVar('T')

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
