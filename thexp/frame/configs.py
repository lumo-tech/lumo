"""
thexp provides an easy git-like config system.

globals : can be used in all repository.
repository : only used in each repository.
running : only used in current run-time.

A global instance is exposed, here is some examples:

```python
from thexp import globs
globs.add_value('dataset','path/to/dataset/root/',globs.LEVEL.globals)
root = globs['dataset']
```

you can get all config by use `globs.items()`
"""
from pprint import pformat
from typing import Any

from thexp.utils.paths import global_config, write_global_config
from ..globals import _CONFIGL, _GITKEY


class Globals:
    LEVEL = _CONFIGL

    def __init__(self):
        self._configs = [
            Config(_CONFIGL.running),
            Config(_CONFIGL.repository),
            Config(_CONFIGL.globals),
        ]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, item):
        for config in self._configs:
            if item in config:
                return config[item]

    def __setitem__(self, key, value):
        self._configs[0][key] = value

    def add_value(self, key, value, level=_CONFIGL.globals):
        if level == _CONFIGL.globals:
            self._configs[2][key] = value
        elif level == _CONFIGL.repository:
            self._configs[1][key] = value
        elif level == _CONFIGL.running:
            self._configs[0][key] = value
        else:
            assert False, 'level name error {}'.format(level)

    def get_value(self, key, level=_CONFIGL.globals, default=None):
        try:
            if level == _CONFIGL.globals:
                return self._configs[2][key]
            elif level == _CONFIGL.repository:
                return self._configs[1][key]
            elif level == _CONFIGL.running:
                return self._configs[0][key]
        except:
            return default
        assert False, 'level name error {}'.format(level)

    def items(self):
        return {
            _CONFIGL.globals: self._configs[2].items(),
            _CONFIGL.repository: self._configs[1].items(),
            _CONFIGL.running: self._configs[0].items(),
        }

    def __repr__(self):
        return "Globals({})".format(pformat({
            _CONFIGL.globals: self._configs[2].items(),
            _CONFIGL.repository: self._configs[1].items(),
            _CONFIGL.running: self._configs[0].items(),
        }))

    @property
    def runtime_config(self):
        return self._configs[0]

    @property
    def repository_config(self):
        return self._configs[1]

    @property
    def globals_config(self):
        return self._configs[2]

    @property
    def repo_root(self):
        from ..utils.repository import git_root
        return git_root()


class Config:
    """
    试验配置，根据等级分为用户级（整个用户），repo级（当前项目），实验级（当次运行）
    """
    config_levels = {_CONFIGL.running, _CONFIGL.repository, _CONFIGL.globals}

    def __init__(self, config_level):
        assert config_level in Config.config_levels, 'config level must in {}'.format(Config.config_levels)
        self._config_level = config_level
        if config_level == _CONFIGL.running:
            self._config_dict = {}
        elif config_level == _CONFIGL.repository:
            self._repo = None
            self._config_dict = None
        elif config_level == _CONFIGL.globals:
            self._config_dict = global_config()

    @property
    def repo(self):
        if self.config_level == _CONFIGL.repository:
            from ..utils.repository import load_repo
            self._repo = load_repo()
        return self._repo

    @property
    def repo_config(self):
        if not self.repo_level or self.repo is None:
            return {}
        if self._config_dict is None:
            from ..utils.repository import git_config
            self._config_dict = git_config(self.repo)
        return self._config_dict

    @property
    def config_level(self):
        return self._config_level

    @property
    def running_level(self):
        return self._config_level == _CONFIGL.running

    @property
    def globals_level(self):
        return self._config_level == _CONFIGL.globals

    @property
    def repo_level(self):
        return self._config_level == _CONFIGL.repository

    def __setitem__(self, key, value: str):
        """
        key 和 value尽量简洁
        :param key:
        :param value:
        :return:
        """
        key = str(key)
        if self._config_level == _CONFIGL.globals:
            self._config_dict[key] = value
            write_global_config(self._config_dict)
        elif self._config_level == _CONFIGL.repository:
            from thexp.utils.repository import git_config_syntax
            value = git_config_syntax(value)
            repo = self.repo
            if repo is not None:
                writer = repo.config_writer()
                writer.add_value(_GITKEY.section_name,key,value)
                writer.write()
                writer.release()
            self._config_dict[key] = value
        elif self._config_level == _CONFIGL.running:
            self._config_dict[key] = value

    def __getitem__(self, key):
        key = str(key)
        if self._config_level in {_CONFIGL.globals, _CONFIGL.running}:
            if key not in self._config_dict:
                raise AttributeError(key)
            return self._config_dict[key]
        elif self._config_level == _CONFIGL.repository:
            if key not in self.repo_config:
                raise AttributeError(key)
            return self.repo_config[key]

    def items(self):
        if self._config_level == _CONFIGL.running:
            return self._config_dict.items()
        elif self._config_level == _CONFIGL.repository:
            return self.repo_config.items()
        elif self._config_level == _CONFIGL.globals:
            return self._config_dict.items()

    def __contains__(self, item):
        try:
            _ = self[item]
            return True
        except:
            return False

    def __repr__(self) -> str:
        return pformat(self.items())


globs = Globals()
