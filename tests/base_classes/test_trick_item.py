"""

"""

from lumo.base_classes.trickitems import NoneItem, AvgItem


def test_none_item():
    none = NoneItem()
    assert none > 0
    assert none >= 0
    assert none < 0
    assert none <= 0
    assert none != 0
    assert none == None
    assert none is not None
    assert none / 2 == 0.5
    assert none * 2 == 2
    assert none + 1 == 1
    assert none - 1 == -1


def test_avg_item():
    avg = AvgItem()
    avg.update(4)
    avg.update(3)
    avg.update(2)
    assert avg._item == 2
    assert avg.avg == 3
