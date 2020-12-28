"""

"""
import sys
sys.path.insert(0,"../")
from thexp import __VERSION__
print(__VERSION__)

from thexp import Experiment,globs

exp = Experiment("expname")

# glob.add_value("key",'value','user')
# glob.add_value("key",'value','exp')
# glob.add_value("key",'value','repository')
globs['a'] = 4

from pprint import pprint
pprint(globs.items())
pprint(exp.config_items())