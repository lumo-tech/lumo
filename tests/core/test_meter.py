from collections import OrderedDict

import pytest

from lumo.core.meter import ReduceItem, Meter


def test_meter():
    m = Meter()
    m['loss'] = 0.5
    m['accuracy'] = 0.8

    # test __getitem__ method
    assert m['loss'] == 0.5

    # test __setitem__ method
    m['loss'] = 0.2
    assert m['loss'] == 0.2

    # test __repr__ method
    assert repr(m) == 'loss: 0.2 | accuracy: 0.8'

    # test keys method
    assert set(m.keys()) == {'loss', 'accuracy'}

    # test todict method
    assert m.todict() == {'loss': 0.2, 'accuracy': 0.8}

    # test sorted method
    sorted_m = m.sorted()
    assert isinstance(sorted_m, Meter)
    assert set(sorted_m.keys()) == {'accuracy', 'loss'}
    assert repr(sorted_m) == 'accuracy: 0.8 | loss: 0.2'

    # test update method
    m.update({'loss': 0.1, 'precision': 0.9})
    assert set(m.keys()) == {'loss', 'accuracy', 'precision'}
    assert m.todict() == {'loss': 0.1, 'accuracy': 0.8, 'precision': 0.9}

    # test from_dict method
    m2 = Meter.from_dict(OrderedDict([('loss', 0.1), ('accuracy', 0.8), ('precision', 0.9)]))
    assert set(m2.keys()) == {'loss', 'accuracy', 'precision'}
    assert m2.todict() == {'loss': 0.1, 'accuracy': 0.8, 'precision': 0.9}

    # test scalar_items method
    m3 = Meter()
    m3['loss'] = 0.5
    m3['accuracy'] = '80%'
    m3['precision'] = [0.9, 0.8]
    assert set(m3.scalar_items()) == {('loss', 0.5), ('accuracy', '80%')}


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
