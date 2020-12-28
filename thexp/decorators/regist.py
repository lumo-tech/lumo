from typing import Union, Dict, List, Callable


def regist_func(val: Union[Dict[str, Callable], List[Callable]], name_=None):
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
