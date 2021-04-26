"""

"""
import sys
sys.path.insert(0,"../")
from lumo import __version__
print(__version__)

from lumo import Experiment,globs

exp = Experiment("expname")

globs['a'] = 4

from pprint import pprint
pprint(globs.items())
pprint(exp.config_items())