""""

"""

from typing import Any




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
                if not key.startswith("_"):
                    continue
                if isinstance(value, set):
                    v = attrs.setdefault(key, set())
                    v.update(value)
                elif isinstance(value, dict):
                    v = attrs.setdefault(key, dict())
                    v.update(value)

        return type.__new__(cls, name, bases, dict(attrs))


class meta_attr(type):
    """用于记录所有attr子类的类名和类信息，用于序列化和反序列化"""

    def __new__(cls, *args: Any, **kwargs: Any):
        from lumo.base_classes.attr import _attr_clss
        cls = type.__new__(cls, *args, **kwargs)
        _attr_clss[cls.__name__] = cls
        return cls



