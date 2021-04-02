"""

"""

from lumo.utils.timing import TimeIt
import time

if __name__ == '__main__':
    t = TimeIt()
    t.start()
    for i in range(10):
        time.sleep(0.1)
        t.mark(i)

    t.end()
    print(t.meter())
from torch.utils.data.dataloader import DataLoader

from torch.nn import Module