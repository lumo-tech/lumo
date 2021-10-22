from lumo.proc.path import libhome
import os
from lumo.utils.safe_io import IO
from lumo.base_classes import tree
from lumo.utils.filebranch import FileBranch

import time


def get_heartbeats():
    pass


def get_test_info(test_root):
    pass


class PathView:
    def __init__(self, root, **metainfo):
        self._root = root
        self.metainfo = metainfo

    @property
    def root(self):
        return self._root

    def tags(self):
        return sorted(self.metainfo.keys())

    def views(self):
        return []

    def get(self, tag, default=None):
        return self.metainfo.get(tag, default=default)

    def get_view(self, view: str):
        func = getattr(self, view)
        if callable(func):
            return func()
        return None

    @property
    def exists(self):
        return os.path.exists(self.root)

    def view(self) -> dict:
        return {}

    def jsonify(self):
        res = {}
        res['info'] = self.view()
        res['view'] = self.__class__.__name__.lower().replace('view', '')
        for k, v in self.metainfo.items():
            if isinstance(v, PathView):
                res[k] = v.view()
            else:
                res[k] = v
        return res

    def update(self, **kwargs):
        metainfo = {k: v for k, v in self.metainfo.items()}
        metainfo.update(kwargs)
        return self.__class__(self.root, **metainfo)

    def __str__(self):
        return f'{self.__class__.__name__}({self.root})'

    __repr__ = __str__


class FileView(PathView):

    def view(self):
        content = ''
        if self.exists:
            content = IO.load_text(self.root)

        return {
            'isfile': True,
            'path': self.root,
            'exists': self.exists,
            'content': content
        }


class DirectoryView(PathView):
    def view(self):
        return {
            'isfile': False,
            'path': self.root,
            'exists': self.exists,
        }


class TestView(DirectoryView): pass


class ExpView(DirectoryView): pass


class LogView(FileView): pass


class TestInfoView(DirectoryView): pass


class JsonView(FileView):


class AttrView(JsonView): pass


class ParamsView(AttrView): pass


class Root:
    def __init__(self):
        self._root = tree()
        self._fb = FileBranch(libhome(), touch=False)

    def get_heartbeats(self, fb=None):
        if fb is None:
            fb = self._fb
        cur = time.time()
        for f in fb.branch('heartbeat').find_file_in_depth('.hb'):
            if cur - os.stat(f).st_mtime < 10:
                val = TestView(IO.load_text(f), activate=True)
                os.remove(f)
                yield val

    def get_activity(self):
        for f in list(sorted(self._fb.branch('diary').find_file_in_depth('.log'))):
            for line in IO.load_text(f).split('\n'):
                date = os.path.basename(f).split('.')[0]

                res = line.split(', ', maxsplit=1)
                if len(res) == 2:
                    start_time, test_root = res
                    yield TestView(test_root, start_time=start_time, start_date=date)

    def get_experiments(self):
        exp_fb = self._fb.branch('experiment')
        return [ExpView(os.path.join(exp_fb.root, f)) for f in exp_fb.listdir()]

    def get_experiment(self, exp_name):
        exp_fb = self._fb.branch('experiment', exp_name)
        activity = self.get_heartbeats(exp_fb)
        tests = list(exp_fb.find_dir_in_depth('^[0-9]'))
        hist = set()
        [hist.add(i.root) for i in activity]
        yield from activity
        for t in tests:
            if t not in hist:
                yield TestView(t)
                hist.add(t)

    def get_test_info(self, test_root=None, exp_name=None, test_name=None):
        if exp_name is not None and test_name is not None:
            test_fb = self._fb.branch('experiment', exp_name, test_name)
        else:
            test_fb = FileBranch(test_root)

        return TestView(
            test_fb.root,
            info=TestInfoView(test_fb.branch('info').root),
            log=[LogView(i) for i in test_fb.find_file_in_depth('.log$')],
            params=[ParamsView(i) for i in test_fb.find_file_in_depth('^params')],
        )
