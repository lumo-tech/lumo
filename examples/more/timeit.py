"""

"""

from lumo.utils import Timer
import time

if __name__ == '__main__':
    t = Timer()
    t.start()
    for i in range(10):
        time.sleep(0.1)
        t.mark(i)

    t.end()
    print(t.meter())

