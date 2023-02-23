from lumo.proc import glob


class ExpHook:
    name = None  # type: str
    configs = {}

    def __new__(cls):
        if cls.name is None:
            cls.name = cls.__name__
        self = super().__new__(cls)
        return self

    @property
    def config_name(self):
        return f'HOOK_{self.name.upper()}'

    @property
    def config_string(self):
        return ', '.join(f'{k}={glob.get(k, v)}' for k, v in self.configs.items())

    def regist(self, exp): self.exp = exp

    def on_start(self, exp, *args, **kwargs): pass

    def on_end(self, exp, end_code=0, *args, **kwargs): pass

    def on_progress(self, exp, step, *args, **kwargs): pass

    def on_newpath(self, exp, *args, **kwargs): pass

    def __repr__(self):
        return f"Hook(name={self.__class__.name}, switch={self.config_name}, {self.config_string})"
