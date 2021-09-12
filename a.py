# from lumo.kit.meter import AvgMeter, Meter
# from lumo import Logger
# import numpy as np
# import torch
# import time
#
# from lumo.calculate.schedule import CosSchedule
#
# log = Logger()
# start = time.time()
# avg = AvgMeter()
# a = np.linspace(0, 101, 100)
# b = np.linspace(0, 101, 1000)
# c = np.linspace(0, 10, 10000)
# d = np.linspace(0, 1, 100000)
# e = np.linspace(0, 0.1, 1000000)
# sche = CosSchedule(1e-6, 1e-6, 0, 50)
# for i in range(10):
#     m = Meter()
#     m.a = a[i]
#     m.b = b[i]
#     m.c = c[i]
#     m.d = d[i]
#     m.e = e[i]
#
#     m.lr = sche(i)
#
#     avg.update(m)
#     log.info(avg)
#
# print(list(avg.items()))
#
# # import numpy as np
# # print(np.ceil(np.log10((1 / 2))))
# # max(8, int(1 / (1e-4 + 1e-10)))

from lumo.proc.date import strftime
print(strftime('%m%d'))
import shutil
shutil.copy()