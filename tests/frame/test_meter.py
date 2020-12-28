"""

"""

from thexp.frame.meter import Meter, AvgMeter
import torch

from thexp import Meter, AvgMeter


def test_meter():
    meter = Meter()
    meter.a = 1
    meter.b = 0.5
    meter.c = 2.5
    meter.percent(meter.b_)
    meter.int(meter.c_)
    meter.float(meter.a_)


def test_avgmeter():
    avg = AvgMeter()

    avg.average(avg.a_)
    avg.a = 1
    avg.a = 2
    avg.a = 3
    assert avg.a.avg == 2  # AvgItem object
    assert avg.a.item == 3

    avg.b = 1
    avg.b = 2
    assert avg.b == 2  # int object

    avg.average(avg.c_)
    x, y = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
    avg.c = x
    avg.c = y
    assert (avg.c.avg == (x + y) / 2).all()
    assert (avg.c.item == y).all()

    assert avg.meter.a == 3
    assert avg.avg.a == 2
    assert avg.meter.b == 2
    assert avg.avg.b == 2
