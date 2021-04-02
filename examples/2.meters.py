"""

"""
import sys
sys.path.insert(0,"../")
from lumo import __version__
print(__version__)


from lumo import Meter,AvgMeter
import torch

m = Meter()
m.a = 1
m.b = "2"
m.c = torch.rand(1)[0]

m.c1 = torch.rand(1)
m.c2 = torch.rand(2)
m.c3 = torch.rand(4, 4)
print(m)

m = Meter()
m.a = 0.236
m.b = 3.236
m.c = 0.23612312
m.percent(m.a_)
m.int(m.b_)
m.float(m.c_,2)
print(m)


am = AvgMeter()
for j in range(5):
    for i in range(100):
        m = Meter()
        m.percent(m.c_)
        m.a = 1
        m.b = "2"
        m.c = torch.rand(1)[0]

        m.c1 = torch.rand(1)
        m.c2 = torch.rand(2)
        m.c3 = torch.rand(4, 4)
        m.d = [4]
        m.e = {5: "6"}
        # print(m)
        am.update(m)
    print(am)

from lumo import Meter
m = Meter()
print(m.k)

m.all_loss = m.all_loss + 5
m.all_loss = m.all_loss + 3
print(m.all_loss)