import glob
import os
from lumo.utils import re


def foo(*_, **__): pass


class FileBranch:
    """
    A directory manager.

    .lumo/experiment/<experment-name>/<test-name>/
    .lumo/backup/<experiment-name>/.backend
    .lumo/backup/<experiment-name>/...
    .lumo/blob/<experiment-name>/<test-name>/

    """

    def __init__(self, root, touch=False, listener=None):
        self._root = os.path.expanduser(root)
        self._cache = set()
        if listener is None:
            listener = foo
        self._listener = listener
        if touch:
            self.makedir(root)

    def __str__(self):
        return f'{self.__class__.__name__}({self.root})'

    @property
    def root(self):
        return os.path.abspath(self._root)

    exists = os.path.exists

    def send(self, *path):
        self._listener(os.path.join(*path))

    def file(self, fn, *dirs):
        fdir = self.makedir(*dirs)
        fn = os.path.join(fdir, fn)
        self.send(fn)
        return fn

    def makedir(self, *dirs):
        fdir = os.path.join(self.root, *dirs)
        os.makedirs(fdir, exist_ok=True)
        self.send(fdir)
        return fdir

    def branch(self, *name):
        res = FileBranch(os.path.join(self.root, *name), touch=True, listener=self._listener)
        self.send(res.root)
        return res

    @property
    def parent(self):
        res = FileBranch(os.path.dirname(self.root), listener=self._listener)
        self.send(res.root)
        return res

    def listdir(self):
        return os.listdir(self.root)

    def walk(self):
        yield from os.walk(self.root)

    def finddir(self, regex, depth=-1):
        match = re.compile(regex)
        for f in self.listdir():
            if os.path.isdir(f):
                self.branch(f)

    def find_dir_in_depth(self, regex, depth=0):
        if isinstance(regex, str):
            match = re.compile(regex)
        else:
            match = regex
        for f in self.listdir():
            if os.path.isdir(os.path.join(self.root, f)):
                if depth == 0:
                    if match.search(f) is not None:
                        yield os.path.join(self.root, f)
                else:
                    yield from self.branch(f).find_dir_in_depth(match, depth - 1)

    def find_file_in_depth(self, regex, depth=0):
        if isinstance(regex, str):
            match = re.compile(regex)
        else:
            match = regex
        for f in self.listdir():
            if os.path.isfile(os.path.join(self.root, f)):
                if depth == 0:
                    if match.search(f) is not None:
                        yield os.path.join(self.root, f)
            else:
                yield from self.branch(f).find_file_in_depth(match, depth - 1)
