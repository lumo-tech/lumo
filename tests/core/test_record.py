import pytest

from lumo import Record, Meter


@pytest.fixture
def record():
    return Record(stage='test')


def test_stage(record):
    assert record.stage == 'test'


def test_record(record):
    for i in range(10):
        m = Meter()
        m.sum.C = 512
        record.record(m)
    record.record({'loss': 0.5, 'accuracy': 0.8})
    assert record._agg['loss'].res == 0.5
    assert record._agg['accuracy'].res == 0.8


def test_record_meter(record):
    for i in range(10):
        m = Meter()
        m.sum.C = 512
        record.record(m)
    assert record.agg()['C'] == 512 * 10
    # assert record._agg['accuracy'].res == 0.8


def test_clear(record):
    record.record({'loss': 0.5, 'accuracy': 0.8})
    record.clear()
    assert len(record._agg) == 0
    assert len(record._cache) == 0


def test_flush(record):
    record.record({'loss': 0.5, 'accuracy': 0.8})
    record.flush()
    assert len(record._cache) == 0


def test_str(record):
    record.record({'loss': 0.5, 'accuracy': 0.8})
    assert str(record) == 'loss=0.5, accuracy=0.8'
