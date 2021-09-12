from lumo.calculate.schedule import *
from lumo import Logger
import numpy as np

learning_rate = 2e-4
epoches_stage1 = 5
epoches_stage2 = 5
epoches_stage3 = 20
train_dataloader = [200] * 200
lr_scheduler = SchedulerList(
    [
        CosScheduler(start=learning_rate, end=1e-6, left=0, right=epoches_stage1),
        CosScheduler(start=learning_rate, end=1e-6, left=epoches_stage1,
                     right=(epoches_stage1 + epoches_stage2)),
        CosScheduler(start=learning_rate, end=1e-6, left=(epoches_stage1 + epoches_stage2),
                     right=(epoches_stage1 + epoches_stage2 + epoches_stage3))
    ]
)

# # epoch = 44
# idx = np.linspace(-1, 40, num=20000)
# from matplotlib import pyplot as plt
#
# print()
# print(lr_scheduler(0))
# plt.plot([lr_scheduler(i) for i in idx])
# plt.show()
#
# log = Logger()
# from lumo import Meter, AvgMeter
#
# avg = AvgMeter()
#
# for i in idx:
#     # print(i,lr_scheduler(i))
#     avg.lr = lr_scheduler(i)
#     log.inline(avg)

cos = CosScheduler(start=learning_rate, end=1e-6, left=2,
                   right=2)
c = ConstantScheduler(4)
print(c(0))

print(cos(1))

cos.plot()