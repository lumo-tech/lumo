"""

"""
import sys
sys.path.insert(0,"../")
from thexp import __VERSION__
print(__VERSION__)
import time

from thexp import Experiment

exp = Experiment("expname")
print(exp.make_exp_dir("explevel"))
print(exp.makedir("testlevel"))

from thexp import Params

params = Params()
exp.add_plugin("params", dict(
    _param_hash = params.hash(),
    data = params.inner_dict.jsonify()
))

from thexp import Logger

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


