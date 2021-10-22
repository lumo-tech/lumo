import os

from . import re


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
            self._listeners = []
        elif callable(listener):
            self._listeners = [listener]
        else:
            self._listeners = listener

        if touch:
            self.makedir(root)

    def __str__(self):
        return f'{self.__class__.__name__}({self.root})'

    @property
    def root(self) -> str:
        return os.path.abspath(self._root)

    exists = os.path.exists

    @property
    def isfile(self):
        return os.path.isfile(self.root)

    @property
    def isdir(self):
        return os.path.isdir(self.root)

    def send(self, *path):
        for l in self._listeners:
            l(os.path.join(*path))

    def file(self, fn, *dirs) -> str:
        fdir = self.makedir(*dirs)
        fn = os.path.join(fdir, fn)
        self.send(fn)
        return fn

    def makedir(self, *dirs):
        fdir = os.path.join(self.root, *dirs)
        if not os.path.exists(fdir):
            os.makedirs(fdir, exist_ok=True)
        self.send(fdir)
        return fdir

    def branch(self, *name):
        res = FileBranch(os.path.join(self.root, *name), touch=True, listener=self._listeners)
        self.send(res.root)
        return res

    def parent(self, level=1):
        res = self
        if level > 0:
            res = FileBranch(os.path.dirname(res.root), listener=res._listeners)
            res.send(res.root)
            return res.parent(level - 1)
        return res

    def listdir(self, abs=False):
        res = os.listdir(self.root)
        if abs:
            res = [os.path.join(self.root, f) for f in res]
        return res

    def walk(self):
        yield from os.walk(self.root)

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


def touch(file, content=None):
    if content is None:
        content = ''
    f = open(file, 'w', encoding='utf8')
    f.write(content)
    f.close()
