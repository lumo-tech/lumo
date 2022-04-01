from abc import ABC, ABCMeta
from collections import OrderedDict

__all__ = ['PropVar', 'OrderedPropVar', 'AttrPropVar', 'ABCPropVar']

from typing import List
from functools import wraps


def make_dicts(type_name, names: List[str], dic_type=dict):
    def outer(func):
        @wraps(func)
        def inner(cls, *args, **kwargs):
            self = func(cls)
            for name in names:
                object.__setattr__(self, name, dic_type())
                # setattr(self, name, dic_type())
            return self

        return inner

    type_name.__new__ = outer(type_name.__new__)


def make_dict(type_name, name: str, default):
    def outer(func):
        @wraps(func)
        def inner(cls, *args, **kwargs):
            self = func(cls)
            setattr(self, name, default)
            return self

        return inner

    type_name.__new__ = outer(type_name.__new__)


class PropVar(type):
    """
    """

    def __new__(cls, name, bases, attrs: dict, **kwds):
        clazz = type.__new__(cls, name, bases, dict(attrs))
        make_dicts(clazz, ['_prop'])
        return clazz


class OrderedPropVar(type):
    """
    """

    def __new__(cls, name, bases, attrs: dict, **kwds):
        clazz = type.__new__(cls, name, bases, dict(attrs))
        make_dicts(clazz, ['_prop'], OrderedDict)
        return clazz


class AttrPropVar(type):
    """
    """

    def __new__(cls, name, bases, attrs: dict, **kwds):
        from .attr import Attr
        clazz = type.__new__(cls, name, bases, dict(attrs))
        make_dicts(clazz, ['_prop'], Attr)
        return clazz


class ABCPropVar(ABCMeta):

    def __new__(cls, name, bases, attrs: dict, **kwds):
        from .attr import Attr
        clazz = type.__new__(cls, name, bases, dict(attrs))
        make_dicts(clazz, ['_prop', '_content'], Attr)
        return clazz


class Merge(type):
    """
    元类，用于将子类和父类共有字典，集合时，子类的覆盖行为改为合并父类的字典，集合

    由于用途特殊，仅识别类变量中以下划线开头的变量
    ::
        class A(metaclass=Merge):
            _dicts = {"1": 2, "3": 4}

        class B(A):
            _dicts = {"5":6,7:8}

        print(B._dicts)

    result:
    >>> {'5': 6, '3': 4, '1': 2, 7: 8}
    """

    def __new__(cls, name, bases, attrs: dict, **kwds):
        for base in bases:
            for key, value in base.__dict__.items():  # type:(str,Any)
                if key.endswith("__"):
                    continue
                if isinstance(value, set):
                    v = attrs.setdefault(key, set())
                    v.update(value)
                elif isinstance(value, dict):
                    v = attrs.setdefault(key, dict())
                    v.update(value)

        return type.__new__(cls, name, bases, dict(attrs))
