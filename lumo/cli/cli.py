import fire

doc = """
Usage:
# create templete directory 
lumo init

# easier way to open tensorboard 
lumo board [--logdir=<logdir>]
lumo board [--test=<test_name>] # find test_name and tensorboard it 
lumo board  # default open ./board

# restore code snapshot of some test
lumo reset <test_name>


# archive code snapshot of some test
lumo archive <test_name>

# delete some test directly
lumo delete <test_name>

# print log file
lumo log <test_name>

# print params of this test
lumo params <test_name>

# <test_name>/--test=<test_name>/--test_name=<test_name>

# TODO
lumo config user --k=v
lumo config repo --k=v
"""
import sys
from lumo import __version__

from lumo.decorators import regist_func

func_map = {}


@regist_func(func_map)
def init():
    import shutil
    import os

    templete_dir = os.path.join(os.path.dirname(__file__), 'templete')
    src_dir = os.getcwd()
    from lumo.utils.repository import init_repo

    dir_name = os.path.basename(src_dir)
    init_repo(src_dir)
    shutil.copytree(templete_dir, os.path.join(src_dir, dir_name))
    # os.rename(os.path.join(src_dir, 'templete'), os.path.join(src_dir, dir_name))
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py-tpl'):
                nfile = "{}.py".format(os.path.splitext(file)[0])
                os.rename(os.path.join(root, file), os.path.join(root, nfile))


@regist_func(func_map)
def check(*args, **kwargs):
    from lumo.utils.repository import init_repo
    init_repo()

#
# def _board_with_logdir(logdir, *args, **kwargs):
#     import os
#     import subprocess
#
#     tmpdir = os.path.join(os.path.dirname(logdir), 'board_tmp')
#     os.makedirs(tmpdir, exist_ok=True)
#     try:
#         subprocess.check_call(['tensorboard', '--logdir={}'.format(logdir),
#                                *['--{}={}'.format(k, v) for k, v in kwargs.items()]],
#                               env=dict(os.environ, TMPDIR=tmpdir))
#     except KeyboardInterrupt as k:
#         print('bye.')
#         exit(0)

#
# def _board_with_test_name(test_name, *args, **kwargs):
#     query = Q.tests(test_name)
#     if query.empty:
#         raise IndexError(test_name)
#     vw = query.to_viewer()
#     if not vw.has_board():
#         raise AttributeError('{} has no board or has been deleted'.format(test_name))
#     else:
#         _board_with_logdir(vw.board_logdir, **kwargs)
#
#
# @regist_func(func_map)
# def board(*args, **kwargs):
#     if len(args) > 0:
#         kwargs.setdefault('test_name', args[0])
#
#     if 'logdir' in kwargs:
#         _board_with_logdir(**kwargs)
#     elif 'test' in kwargs:
#         _board_with_test_name(kwargs['test'])
#     elif 'test_name' in kwargs:
#         _board_with_test_name(**kwargs)
#     else:
#         _board_with_logdir('./board')
#
#
# def _find_test_name(*args, **kwargs):
#     if len(args) > 0:
#         return args[0]
#     elif 'test' in kwargs:
#         return kwargs['test']
#     elif 'test_name' in kwargs:
#         return kwargs['test_name']
#     return None
#
#
# @regist_func(func_map)
# def reset(*args, **kwargs):
#     test_name = _find_test_name(*args, **kwargs)
#     query = Q.tests(test_name)
#     if query.empty:
#         print("can't find test {}".format(test_name))
#         exit(1)
#     exp = query.to_viewer().reset()
#     print('reset from {} to {}'.format(exp.plugins['reset']['from'], exp.plugins['reset']['to']))
#
#
# @regist_func(func_map)
# def archive(*args, **kwargs):
#     test_name = _find_test_name(*args, **kwargs)
#     query = Q.tests(test_name)
#     if query.empty:
#         print("can't find test {}".format(test_name))
#         exit(1)
#     exp = query.to_viewer().archive()
#
#     print('archive {} to {}'.format(test_name, exp.plugins['archive']['file']))
#
#
# @regist_func(func_map)
# def delete(*args, **kwargs):
#     test_name = _find_test_name(*args, **kwargs)
#     query = Q.tests(test_name)
#     if query.empty:
#         print("can't find test {}".format(test_name))
#         exit(1)
#
#     query.to_viewer().delete()
#     print('success delete {}.'.format(test_name))
#
#
# @regist_func(func_map)
# def log(*args, **kwargs):
#     test_name = _find_test_name(*args, **kwargs)
#     query = Q.tests(test_name)
#     if query.empty:
#         print("can't find test {}".format(test_name))
#         exit(1)
#
#     vw = query.to_viewer()
#     if not vw.has_log():
#         print("can't find log file of [{}]".format(test_name))
#         exit(1)
#     print(vw.log_fn)
#
#
# @regist_func(func_map)
# def params(*args, **kwargs):
#     test_name = _find_test_name(*args, **kwargs)
#     query = Q.tests(test_name)
#     if query.empty:
#         print("can't find test {}".format(test_name))
#         exit(1)
#
#     vw = query.to_viewer()
#     p = vw.params
#     if p is not None:
#         print(p)
#     else:
#         print("can't find param object of [{}]".format(test_name))
#         exit(1)
#
#
# @regist_func(func_map)
# def last(*args,**kwargs):
#     N = args[0]
#
#     repos = Q.repos()
#     frepo = kwargs.get('repo',None)
#     if repos is not None:
#         repos = repos[frepo]
#
#     exps = repos.exps()
#     fexp = kwargs.get('exp',None)
#     if fexp is not None:
#         exps = exps[fexp]
#
#     tests = exps.tests()
#     tests






def main(*args, **kwargs):
    # print(args, kwargs)
    if len(args) == 0 or 'help' in kwargs:
        print(doc)
        return

    branch = args[0]
    if branch in func_map:
        func_map[branch](*args[1:], **kwargs)
    else:
        print(doc)

# Fire 不能嵌套在 __main__ 判断里，否则 sys.argv 识别会出问题，目前原因未知
fire.Fire(main)
exit(0)
