from lumo.proc.path import libhome
import os
from lumo.utils.safe_io import IO
from lumo.base_classes import tree
from lumo.utils.filebranch import FileBranch

import time


def get_heartbeats():
    cur = time.time()
    res = []
    for f in FileBranch(libhome()).branch('heartbeat').find_file_in_depth('hb'):
        if cur - os.stat(f).st_mtime > 10:
            test_root = IO.load_text(f)
            res.append(test_root)
        else:
            os.remove(f)

    return res


def get_experiments():
    return list(FileBranch(libhome()).branch('experiment').find_dir_in_depth('.'))


def get_experiment_names():
    return list(FileBranch(libhome()).branch('experiment').listdir('.'))


def get_tests_in_experiment(exp_name):
    return list(FileBranch(libhome()).branch('experiment', exp_name).find_dir_in_depth('^[0-9]'))


def get_test_tags(test_root):
    return set(FileBranch(test_root).branch('tag').listdir())


def filter_tests_with_tag(test_roots: list, *tag):
    res = []
    for test in test_roots:
        test_tags = get_test_tags(test)
        if all([i in test_tags for i in tag]):
            res.append(test)
    return res
