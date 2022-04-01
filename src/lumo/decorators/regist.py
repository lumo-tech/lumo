from collections import OrderedDict
from functools import partial
from typing import Union, Dict, List, Callable


def regist_func_to(val: Union[Dict[str, Callable], List[Callable]], name_=None):
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


class Register():
    def __init__(self, name):
        self.name = name
        self.source = OrderedDict()

    def __str__(self):
        inner = str([(k, v) for k, v in self.source.items()])
        return f"Register({self.name}{inner})"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.source.get(item, None)

    def __call__(self, wrapped, name=None):
        if name is None:
            name = wrapped.__name__
        assert name is not None
        self.source[name] = wrapped
        return wrapped

    def regist(self, name=None):
        return partial(self, name=name)
