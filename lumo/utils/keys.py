from typing import Mapping, TypeVar, Type

T = TypeVar('T')


class K:
    @staticmethod
    def join(*args):
        return '_'.join([i for i in args if i is not None])

    @staticmethod
    def get_str(mem: Mapping, key, default=None) -> str:
        return mem.get(key, default)

    @staticmethod
    def get_type(mem: Mapping, key, default=None, *, type: Type[T]) -> T:
        return mem.get(key, default)
