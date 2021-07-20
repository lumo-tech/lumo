import os
from lumo.base_classes.tree import tree
from typing import Callable, Union


def touch(file, content=None):
    if content is None:
        content = ''
    f = open(file, 'w', encoding='utf8')
    f.write(content)
    f.close()


class Template:
    def __init__(self):
        self.tree = tree()

    def add_directory(self, path, *paths):
        paths = [path] + list(paths)
        tree = self.tree
        for path in paths:
            for sub in path.split('/'):
                tree = tree[sub]
        return self

    def add_file(self, path, *paths, info: Union[Callable, str] = None):
        paths = [path] + list(paths)
        pre = self.tree
        tree = self.tree
        for path in paths:
            for sub in path.split('/'):
                pre = tree
                tree = tree[sub]
        pre[sub] = info
        return self

    def create(self, root='./', ignore_if_not_empty=True):
        if os.path.exists(root) and len(os.listdir(root)) and ignore_if_not_empty:
            return False

        for file, info in self.tree.walk():
            file = os.path.join(root, file)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            if callable(info):
                info(file)
            elif isinstance(info, dict):
                os.makedirs(file, exist_ok=True)
            else:
                touch(file, info)
        return True
