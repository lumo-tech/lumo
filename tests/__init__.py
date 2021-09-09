from lumo.kit.meter import Meter2, AvgMeter
from lumo import Logger
import numpy as np
import torch

log =Logger()
# print(m)
# print(avg)

import time
start = time.time()
avg = AvgMeter()

for i in range(100000):
    m = Meter2()
    m.a = i
    m.min.d = i
    m.b = torch.rand([1])
    m.c = np.random.rand(5)

    avg.update(m)
    log.inline(avg)
end = time.time()
