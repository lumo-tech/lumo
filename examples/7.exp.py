"""

"""
import sys
sys.path.insert(0,"../")
from lumo import __version__
print(__version__)
import time

from lumo import Experiment

exp = Experiment("expname")
print(exp.make_exp_dir("explevel"))
print(exp.makedir("testlevel"))

from lumo import Params

params = Params()
exp.add_plugin("params", dict(
    _param_hash = params.hash(),
    data = params.inner_dict().jsonify()
))

from lumo import Logger

logger = Logger()
fn = logger.add_log_dir(exp.makedir('logger'))
exp.add_plugin('logger', dict(
    fn = fn,
))



time.sleep(1)
try:
    raise Exception("dddd")
except:
    pass

exp.end()


