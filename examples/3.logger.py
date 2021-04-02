"""

"""
import sys
sys.path.insert(0,"../")
from lumo import __version__
print(__version__)


import time
from lumo import Logger,Meter,Params
logger = Logger()

meter = Meter()
meter.a = 3.13524635465
meter.b = "long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;"

for i in range(10):
    logger.inline("inline examples",meter)
    time.sleep(0.2)
logger.info(1,2,3,{4:4})
logger.info(meter)

for i in range(10):
    logger.raw(i,inline=True,append=True)
    time.sleep(0.2)

logger.add_log_dir("./")