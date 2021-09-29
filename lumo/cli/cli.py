import fire

doc = """
Usage:
# create templete directory 
lumo init [dir]

# easier way to open tensorboard 
# lumo board [--logdir=<logdir>]
# lumo board [--test=<test_name>] # find test_name and tensorboard it 
# lumo board  # default open ./board

# restore code snapshot of some test
lumo reset <test_name>

# archive code snapshot of some test
lumo archive <test_name>

# print log file
lumo log <test_name>

# print params of this test
lumo params <test_name>

# <test_name>/--test=<test_name>/--test_name=<test_name>

# TODO
lumo config local --k=v
lumo config global --k=v

# get a free port
lumo port

"""
import sys
# from lumo import __version__

from lumo.decorators import regist_func_to
from lumo import globs

func_map = {}


def init(*args, **kwargs):
    pass


@regist_func_to(func_map)
def board(*args, **kwargs):
    pass


def _find_test_name(*args, **kwargs):
    if len(args) > 0:
        return args[0]
    elif 'test' in kwargs:
        return kwargs['test']
    elif 'test_name' in kwargs:
        return kwargs['test_name']
    return None


@regist_func_to(func_map)
def checkout(*args, **kwargs):
    test_name = _find_test_name(*args, **kwargs)
    query = Q.tests(test_name)
    if query.empty:
        print("can't find test {}".format(test_name))
        exit(1)
    exp = query.to_viewer().reset()
    print('reset from {} to {}'.format(exp.plugins['reset']['from'], exp.plugins['reset']['to']))


#
@regist_func_to(func_map)
def archive(*args, **kwargs):
    test_name = _find_test_name(*args, **kwargs)
    query = Q.tests(test_name)
    if query.empty:
        print("can't find test {}".format(test_name))
        exit(1)
    exp = query.to_viewer().archive()

    print('archive {} to {}'.format(test_name, exp.plugins['archive']['file']))


def find(*args, **kwargs):
    pass


def report(*args, **kwargs):
    pass


@regist_func_to(func_map)
def params(*args, **kwargs):
    test_name = _find_test_name(*args, **kwargs)
    query = Q.tests(test_name)
    if query.empty:
        print("can't find test {}".format(test_name))
        exit(1)

    vw = query.to_viewer()
    p = vw.params
    if p is not None:
        print(p)
    else:
        print("can't find param object of [{}]".format(test_name))
        exit(1)


def main(*args, **kwargs):
    # print(args, kwargs)
    print(f"lumo {__version__}")
    if len(args) == 0 or 'help' in kwargs:
        print(doc)
        return

    branch = args[0]
    if branch in func_map:
        func_map[branch](*args[1:], **kwargs)
    else:
        print(doc)


class Board:
    pass


class Main():
    def init(self):
        pass

    def board(self, logdir=None, test_name=None):
        pass


# Fire 不能嵌套在 __main__ 判断里，否则 sys.argv 识别会出问题，目前原因未知
fire.Fire(Main())
# exit(0)
