"""
Help you find the experiments you have done.
"""
# from .experiment import Experiment
from typing import Any

from lumo.proc.path import libhome
from lumo.utils.filebranch import FileBranch


#
# print(list(FileBranch(libhome()).branch('experiment').find_dir_in_depth('.*', 0)))
# print(list(FileBranch(libhome()).branch('experiment').find_dir_in_depth('[0-9.a-z]{13}t$', 1)))


class Finder(FileBranch):

    def __new__(cls) -> Any:
        self = super().__new__(cls)

        def wrap(func):
            def inner(*args, **kwargs):
                func_name = func.__name__
                if func.__name__ not in self._results:
                    res = func(*args, **kwargs)
                    self._results[func_name] = res
                res = self._results[func_name]
                return res

            return inner

        for func in [self.experiments, ]:
            setattr(self, func.__name__, wrap(func))

        return self

    def __init__(self, root=None, touch=False, listener=None):
        if root is None:
            root = libhome()
        super().__init__(root, touch, listener)
        self._results = {

        }

    def experiments(self):
        # print(self.branch('experiments').root)
        return list(self.branch('experiment').find_dir_in_depth('.*', 0))

    def tests(self, exp_prefix=None):
        if exp_prefix is None:
            return

    def refresh(self):
        self._results.clear()


if __name__ == '__main__':
    f = Finder()
    print(f.experiments())
    print(f.experiments())
    f.refresh()
    print(f.experiments())
    print()
