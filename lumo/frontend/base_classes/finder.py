"""
Help you find the experiments you have done.

 - find experiments
 - find tests


Diary view: 访 GitHub 提交墙
 - github commit status

Compare view：对比不同的结果，展示图表
 - 怎么抽象其他项目跑出来的 result 和本项目跑出来的 result
 - 对比时可以选择比一个 metric 的 最值-最值，序列-最值，序列-序列

TestInfo view
 - 展示单个 Test 的信息，包括执行用时，git 提交，大文件存放（空间占用）

"""
# from .experiment import Experiment
import os
from typing import Any

from lumo.proc.path import libhome
from lumo.utils.filebranch import FileBranch
from lumo.utils.safe_io import IO
import datetime


class View:
    pass


class Timeline:
    def __init__(self):
        self.fb = FileBranch(libhome()).branch('diary')

    def _check_fmt(self, date: str):
        assert datetime.datetime.strptime(date, '%y%m%d')

    def calendar(self):
        last = datetime.datetime.now() - datetime.timedelta(days=10)
        return [i for i in self.fb.listdir() if i[:6] >= last.strftime('%y%m%d')]

    def activity(self, date: str):
        self._check_fmt(date)
        logf = self.fb.file(f'{date}.log')
        if not os.path.exists(logf):
            return []
        with open(logf) as r:
            return [i.strip().split(', ', maxsplit=1) for i in r.readlines()]


class TestBlob:
    def __init__(self, test_path):
        self.fb = FileBranch(test_path)

    def tags(self):
        return self.fb.listdir()

    def blob(self, tag):
        fs = self.fb.branch(tag).listdir()
        info_fs_map = {os.path.splitext(i)[0]: i for i in fs if i.endswith('.json')}
        blob_fs = [i for i in fs if not i.endswith('.json')]
        res = []
        for bf in blob_fs:
            info = None
            if bf in info_fs_map:
                info = IO.load_json(self.fb.file(info_fs_map[bf], tag))
            res.append({
                'path': bf,
                'info': info
            })
        return res


class TestInfo:
    def __init__(self, test_path):
        self.fb = FileBranch(test_path)

    def tags(self):
        return self.fb.listdir()

    def tag(self, tag):  # TODO FileBranch 应该抽象为一个 View，可以以统一的接口被前端显示
        return self.fb.branch(tag)

    def logs(self):
        return [i for i in self.fb.listdir() if i.endswith('.log')]

    def params(self):
        return IO.load_json(self.tag('params.json').root)

    def infos(self):
        return self.fb.branch('info').listdir()

    def info(self, fn):
        info_fn = self.fb.branch('info').file(fn)
        if info_fn.endswith('.json'):
            return IO.load_json(info_fn)
        else:
            return IO.load_text(info_fn)

    def blob(self) -> TestBlob:
        return TestBlob(self.info('blob'))


class Finder(FileBranch):

    def __new__(cls) -> Any:
        self = super().__new__(cls)

        def wrap(func):
            def inner(*args, **kwargs):
                func_name = func.__name__
                key = f'{func_name}{args}{kwargs}'
                if key not in self._results:
                    res = func(*args, **kwargs)
                    self._results[key] = res
                res = self._results[key]
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

    def experiments(self, exp_prefix=None):
        if exp_prefix is None:
            exp_prefix = '.*'
        return list(self.branch('experiment').find_dir_in_depth(exp_prefix, 0))

    def tests(self, exp_prefix=None, test_prefix=None):
        for exp in self.experiments(exp_prefix):
            FileBranch(exp).find_dir_in_depth()
        if exp_prefix is None:
            return list(FileBranch(libhome()).branch('experiment').find_dir_in_depth('[0-9.a-z]{13}t$', 1))
        else:

            return list(FileBranch(libhome()).branch('experiment').find_dir_in_depth('[0-9.a-z]{13}t$', 1))

    def refresh(self):
        self._results.clear()


if __name__ == '__main__':
    print(Timeline().calendar())
    print()
    ti = TestInfo(Timeline().activity('211011')[-1][1])
    print(ti.tags())
    print(ti.tag('saver'))
    print(ti.params())
    # for i in ti.infos():
    #     print(ti.info(i))
    #
    # print(ti.blob().tags())
    # print(ti.blob().blob('saver'))
    # print(ti.blobs())
