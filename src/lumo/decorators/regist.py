from collections import OrderedDict
from functools import partial
from typing import Union, Dict, List, Callable


def regist_func_to(val: Union[Dict[str, Callable], List[Callable]], name_=None):
    """
    A decorator function that registers a function to a dictionary or list.

    Args:
        val: A dictionary or list to register the function to.
        name_: The key or index of the function in the dictionary or list. If not provided, it defaults to the name
            of the function.

    Returns:
        The decorated function.

    Notes:
        This decorator function is used to register a function to a dictionary or list. If the `val` argument is a
        dictionary, the function will be added to the dictionary with a key equal to `name_` or the function name if
        `name_` is not provided. If the `val` argument is a list, the function will be appended to the list. The
        registered function can then be retrieved and called later. The `name_` argument is optional, but if it is
        provided it should be a string. The `val` argument should be either a dictionary or a list of functions.
    """

    def wrap(func):
        if name_ is None:
            name = func.__name__
        else:
            name = name_
        if isinstance(val, dict):
            val[name] = func
        elif isinstance(val, list):
            val.append(func)

        return func

    return wrap


class Register:
    """
    A class for registering functions.

    Args:
        name: The name of the register.

    Attributes:
        name: The name of the register.
        source: An ordered dictionary that holds the registered functions.

    Methods:
        __str__(): Returns a string representation of the register.
        __repr__(): Returns a string representation of the register.
        __getitem__(item): Gets a function from the register by name.
        __call__(wrapped, name): A decorator function that adds a function to the register.
        regist(name): A method that returns a partial function of __call__ with the register's name.
    """

    def __init__(self, name: str):
        """
        Initialize the register.

        Args:
            name: The name of the register.
        """
        self.name = name
        self.source = OrderedDict()

    def __str__(self) -> str:
        """
        Return a string representation of the register.

        Returns:
            A string representation of the register.
        """
        inner = str([(k, v) for k, v in self.source.items()])
        return f"Register({self.name}{inner})"

    def __repr__(self) -> str:
        """
        Return a string representation of the register.

        Returns:
            A string representation of the register.
        """
        return self.__str__()

    def __getitem__(self, item: str):
        """
        Get a function from the register by name.

        Args:
            item: The name of the function.

        Returns:
            The function with the given name, or None if the function is not in the register.
        """
        return self.source.get(item, None)

    def __call__(self, wrapped: callable, name: str = None) -> callable:
        """
        Add a function to the register.

        Args:
            wrapped: The function to be added to the register.
            name: The name of the function in the register. If not provided, the function's name will be used.

        Returns:
            The original function, unchanged.

        Raises:
            AssertionError: If the `name` argument is not provided.
        """
        if name is None:
            name = wrapped.__name__
        assert name is not None
        self.source[name] = wrapped
        return wrapped

    def regist(self, name: str = None):
        """
        Returns a partial function of __call__ with the register's name.

        Args:
            name: The name of the function in the register. If not provided, the function's name will be used.

        Returns:
            A partial function of __call__ with the register's name.

        Notes:
            This method is used to create a decorator that will add a function to the register. The `name` argument
            is optional, but if it is provided it should be a string.
        """
        return partial(self, name=name)
