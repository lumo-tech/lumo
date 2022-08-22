import fire
import os
from .functional import *
from ..exp import Experiment

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


class Main:
    def sum(self, tid):
        """

        Args:
            tid: test_name or test_root

        Returns:

        """
        from ..exp.finder import summary_experiment
        summary_experiment(tid)

    def today(self):
        pass

    def init(self, path):
        git_init(path)
        print(os.path.abspath(path))

    def extract(self, test_root, output=None, verbose=True):
        exp = Experiment.from_disk(test_root)
        test_extract(test_root, output=output, verbose=verbose)

    def clone(self, arg: str, alias: str = None):
        """
        if template:
            git clone template_map[arg] alias
        else:
            git clone <arg> alias

        Args:
            arg: url or template name
            template: template id
            alias: alias

        Returns:

        """
        if '/' not in arg:
            _, path = git_clone_from_template(arg, alias)
        else:
            _, path = git_clone(arg, alias)
        git_init(path)


fire.Fire(Main())
exit(0)
