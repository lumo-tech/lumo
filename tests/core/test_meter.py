from lumo.core.meter import ReduceItem


def test_avg_item_mean():
    avg = ReduceItem(gb_method='mean')
    avg.update(4)
    avg.update(3)
    avg.update(2)
    assert avg.last == 2
    assert avg.res == 3


def test_avg_item_sum():
    avg = ReduceItem(gb_method='sum')
    avg.update(4)
    avg.update(3)
    avg.update(2)
    assert avg.last == 2
    assert avg.res == 9


def test_avg_item_max():
    avg = ReduceItem(gb_method='max')
    avg.update(4)
    avg.update(3)
    avg.update(2)
    assert avg.last == 2
    assert avg.res == 4


def test_avg_item_min():
    avg = ReduceItem(gb_method='min')
    avg.update(2)
    avg.update(4)
    avg.update(3)
    assert avg.last == 3
    assert avg.res == 2
