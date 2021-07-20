"""
"""
import os
from collections import OrderedDict
from pprint import pformat
from typing import Any

from lumo.utils import safe_io as io
from lumo.proc.const import CFG
from lumo.proc.path import global_config_path


def default_dict():
    return {CFG.PATH.GLOBAL_EXP: os.path.expanduser("~/.lumo/experiments"),
            CFG.PATH.DATASET: os.path.expanduser("~/.lumo/datasets"),
            CFG.PATH.PRETRAINS: os.path.expanduser("~/.lumo/pretrains"),
            CFG.PATH.CACHE: os.path.expanduser("~/.cache/lumo")}


class Globals:
    RUNTIME = CFG.FIELD.RUNTIME
    GLOBAL = CFG.FIELD.GLOBAL

    def __init__(self):
        self._configs = OrderedDict()
        for level in Config.config_levels:
            self._configs[level] = Config(level)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, item):
        for config in self._configs.values():
            if item in config:
                return config[item]
        return None

    def __setitem__(self, key, value):
        self._configs[CFG.FIELD.RUNTIME][key] = value

    def __repr__(self):
        return "Environ({})".format(pformat(self.items()))

    @property
    def runtime_config(self):
        return self._configs[CFG.FIELD.RUNTIME]

    @property
    def globals_config(self):
        return self._configs[CFG.FIELD.GLOBAL]

    def set(self, key, value, level=CFG.FIELD.GLOBAL):
        self._configs[level][key] = value

    def get(self, key, level=CFG.FIELD.GLOBAL, default=None):
        """"""
        return self._configs[level][key, default]

    def get_first(self, *args, default=None):
        for arg in args:
            res = self[arg]
            if res is not None:
                return res
        return default

    def items(self):
        """return all config, like dict.items()"""
        return {
            CFG.FIELD.GLOBAL: self._configs[CFG.FIELD.GLOBAL].items(),
            CFG.FIELD.RUNTIME: self._configs[CFG.FIELD.RUNTIME].items(),
        }

    def require(self, key, level=CFG.FIELD.GLOBAL):
        """
        Return value if exists, or require user to input value for the specified`key`.

        Then value be typed will be stored in config of the `level`

        Args:
            key:
            level:

        Returns:

        """
        res = self[key]
        if res is None:
            res = input(f'{level} level value `{key}` required.')
            self.add_value(key, res, level)
        return res


class Config:
    """
    Single Level Config
    """
    config_levels = [CFG.FIELD.DEFAULT, CFG.FIELD.RUNTIME, CFG.FIELD.GLOBAL]

    def __init__(self, config_level):
        assert config_level in Config.config_levels, 'config level must in {}'.format(Config.config_levels)
        self._config_level = config_level
        self._config_fn = None
        self._config_dict = None

    @property
    def config_level(self):
        return self._config_level

    @property
    def running_level(self):
        return self.config_level == CFG.FIELD.RUNTIME

    @property
    def globals_level(self):
        return self.config_level == CFG.FIELD.GLOBAL

    @property
    def default_level(self):
        return self.config_level == CFG.FIELD.DEFAULT

    @property
    def config_fn(self):
        if self.globals_level:
            self._config_fn = global_config_path()
            self._config_dict = io.load_json(self._config_fn)
        elif self.running_level:
            self._initial_os_env()
        elif self.default_level:
            self._config_dict = default_dict()
        return self._config_fn

    @property
    def config_dict(self):
        if self._config_dict is not None:
            return self._config_dict

        if self.globals_level:
            self._config_dict = io.load_json(self.config_fn)
        elif self.running_level:
            self._config_dict = {}
            self._initial_os_env()
        elif self.default_level:
            self._config_dict = default_dict()

        if self._config_dict is None:
            self._config_dict = {}

        return self._config_dict

    def _initial_os_env(self):
        for k, v in os.environ.items():
            if k.startswith('LUMO_'):
                self._config_dict[k.lstrip('LUMO_').lower()] = v

    def __setitem__(self, key, value: str):
        key = str(key).lower()
        self.config_dict[key] = value
        self._flush_config()

    def __getitem__(self, key, default=None):
        key = str(key).lower()
        return self.config_dict.get(key, default)

    def __contains__(self, item):
        return item in self.config_dict

    def __repr__(self) -> str:
        return pformat(self.items())

    def _flush_config(self):
        if self._config_fn is not None:
            io.dump_json(self.config_dict, self._config_fn)

    def get(self, key, default=None):
        return self[key, default]

    def items(self):
        return self.config_dict.items()


globs = Globals()
