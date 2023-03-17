from lumo.proc import glob


class BaseExpHook:
    """A base class of hook for experiments that can be registered with an experiment.

    Please Use Exphook in exp.exphook for better typehint.

    """
    name = None  # type: str
    configs = {}

    def __new__(cls):
        if cls.name is None:
            cls.name = cls.__name__
        self = super().__new__(cls)
        return self

    @property
    def config_name(self):
        """Get the configuration name for the hook.

        Returns:
            A string representing the configuration name for the hook.

        """
        return f'HOOK_{self.name.upper()}'

    @property
    def config_string(self):
        """Get the configuration string for the hook.

        Returns:
            A string representing the configuration string for the hook.

        """

        return ', '.join(f'{k}={glob.get(k, v)}' for k, v in self.configs.items())

    def regist(self, exp):
        """Register the hook with an experiment.

        Args:
            exp: The experiment to register the hook with.

        """
        self.exp = exp

    def on_start(self, exp, *args, **kwargs):
        """Execute when the experiment starts.

        Args:
            exp: The experiment that has started.
            *args: Any additional arguments passed to the method.
            **kwargs: Any additional keyword arguments passed to the method.

        """

    def on_end(self, exp, end_code=0, *args, **kwargs):
        """Execute when the experiment ends.

        Args:
            exp: The experiment that has ended.
            end_code (int): The exit code for the experiment.
            *args: Any additional arguments passed to the method.
            **kwargs: Any additional keyword arguments passed to the method.

        """

    def on_progress(self, exp, step, *args, **kwargs):
        """Execute when the experiment makes progress.

        Args:
            exp: The experiment that is making progress.
            step: The current step of the experiment.
            *args: Any additional arguments passed to the method.
            **kwargs: Any additional keyword arguments passed to the method.

        """

    def on_newpath(self, exp, *args, **kwargs):
        """Execute when the experiment creates a new path.

        Args:
            exp: The experiment that is creating a new path.
            *args: Any additional arguments passed to the method.
            **kwargs: Any additional keyword arguments passed to the method.

        """

    def __str__(self):
        """Return a string representation of the hook.

        Returns:
            A string representation of the hook.

        """
        return f"Hook(name={self.__class__.name}, switch={self.config_name}, {self.config_string})"
