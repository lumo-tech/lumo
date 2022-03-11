import fire
from joblib import hash

doc = """
Usage:
# create templete directory 
lumo init [dir]

# easier way to open tensorboard 
# lumo board [--logdir=<logdir>]
# lumo board [--test=<test_name>] # find test_name and tensorboard it 
# lumo board  # default open ./board

# lumo mark <test_name>

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
from lumo.decorators import regist_func_to

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


class Main:
    def init(self):
        pass

    def board(self, logdir=None, test_name=None):
        pass

    def mark(self, test_name=None):
        from lumo.proc.path import libhome
        from lumo.utils import FileBranch
        from lumo.utils import IO

        collection_dir = FileBranch(libhome()).branch('collection')
        if test_name is None:
            for f in collection_dir.listdir(True):
                print(IO.load_text(f))
            return
        from lumo.backend import find_test_by_name

        res = find_test_by_name(test_name)
        if res is None:
            print(f'cannot find {test_name}')
        mark_file = collection_dir.file(hash(res))
        IO.dump_text(res, mark_file)
        print(f'{res} marked.')

        return

    # Fire 不能嵌套在 __main__ 判断里，否则 sys.argv 识别会出问题，目前原因未知


fire.Fire(Main())
# exit(0)
