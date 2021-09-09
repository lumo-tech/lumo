from lumo.kit.meter import Meter2, AvgMeter
from lumo import Logger
import numpy as np
import torch
import time

log = Logger()
start = time.time()
avg = AvgMeter()

for i in range(10):
    m = Meter2()
    m.a = i
    m.min.d = i
    m.b = torch.rand([1])
    m.c = np.random.rand(5)
    m.f = '123'
    m.e = ['123', 3, 5.]

    avg.update(m)
    # log.inline(avg)

print(list(avg.items()))