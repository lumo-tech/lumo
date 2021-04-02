"""

"""


import sys
import time
from lumo import Logger
logger = Logger()
for i in range(1000):
    time.sleep(0.5)
    # sys.stdin.buffer

    # buf = ''
    print('value')
    buf = sys.stdin.read()
    if len(buf) > 0:
        print(buf)
