"""
"""
import os
from collections import OrderedDict
from pprint import pformat
from typing import Any

from lumo.utils import safe_io as io
from lumo.utils.keys import CONFIG
from lumo.utils.paths import local_config_path, global_config_path


class Globals:

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
        self._configs[CONFIG.FIELD.RUNTIME][key] = value

    def __repr__(self):
        return "Environ({})".format(pformat(self.items()))

    @property
    def runtime_config(self):
        return self._configs[CONFIG.FIELD.RUNTIME]

    @property
    def repository_config(self):
        return self._configs[CONFIG.FIELD.REPO]

    @property
    def globals_config(self):
        return self._configs[CONFIG.FIELD.GLOBAL]

    def set(self, key, value, level=CONFIG.FIELD.GLOBAL):
        self._configs[level][key] = value

    def get(self, key, level=CONFIG.FIELD.GLOBAL, default=None):
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
            CONFIG.FIELD.GLOBAL: self._configs[CONFIG.FIELD.GLOBAL].items(),
            CONFIG.FIELD.REPO: self._configs[CONFIG.FIELD.REPO].items(),
            CONFIG.FIELD.RUNTIME: self._configs[CONFIG.FIELD.RUNTIME].items(),
        }

    def require(self, key, level=CONFIG.FIELD.REPO):
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
    config_levels = [CONFIG.FIELD.RUNTIME, CONFIG.FIELD.REPO, CONFIG.FIELD.GLOBAL, ]

    def __init__(self, config_level):
        assert config_level in Config.config_levels, 'config level must in {}'.format(Config.config_levels)
        self._config_level = config_level
        self._config_fn = None
        self._config_dict = {}
        if config_level == CONFIG.FIELD.REPO:
            self._config_fn = local_config_path()
            self._config_dict = io.load_json(self._config_fn)
        elif config_level == CONFIG.FIELD.GLOBAL:
            self._config_fn = global_config_path()
            self._config_dict = io.load_json(self._config_fn)
        else:
            self._initial_os_env()

        if self._config_dict is None:
            self._config_dict = {}

    def _initial_os_env(self):
        for k, v in os.environ.items():
            if k.startswith('LUMO_'):
                self._config_dict[k.lstrip('LUMO_').lower()] = v

    def __setitem__(self, key, value: str):
        key = str(key).lower()
        self._config_dict[key] = value
        self._flush_config()

    def __getitem__(self, key, default=None):
        key = str(key)
        return self._config_dict.get(key, default)

    def __contains__(self, item):
        return item in self._config_dict

    def __repr__(self) -> str:
        return pformat(self.items())

    def _flush_config(self):
        if self._config_fn is not None:
            io.dump_json(self._config_dict, self._config_fn)

    def get(self, key, default=None):
        return self[key, default]

    def items(self):
        return self._config_dict.items()

    @property
    def config_level(self):
        return self._config_level

    @property
    def running_level(self):
        return self._config_level == CONFIG.FIELD.RUNTIME

    @property
    def globals_level(self):
        return self._config_level == CONFIG.FIELD.GLOBAL

    @property
    def repo_level(self):
        return self._config_level == CONFIG.FIELD.REPO


globs = Globals()
