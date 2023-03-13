import os
import pytest
import tempfile

from lumo.exp.metric import Metric


@pytest.fixture
def metric():
    with tempfile.TemporaryDirectory() as tmpdir:
        metric_fn = os.path.join(tmpdir, "test_metric.pkl")
        yield Metric(metric_fn)


def test_value(metric):
    assert isinstance(metric.value, dict)


def test_dump_metric_max(metric):
    key = "test_key_max"
    values = [2, 4, 1, 7, 5]
    expected_value = max(values)
    cur = values[0]
    for value in values:
        max_val = metric.dump_metric(key, value, "max")
        cur = max(value, cur)
        assert max_val == cur

    assert metric.value[key] == expected_value


def test_dump_metric(metric):
    key = "test_key"
    value = 10
    metric.dump_metric(key, value, "max")
    assert metric.value[key] == value


def test_dump_metrics(metric):
    dic = {"key1": 1, "key2": 2, "key3": 3}
    cmp = "min"
    result = metric.dump_metrics(dic, cmp)
    assert result == {"key1": 1, "key2": 2, "key3": 3}


def test_flush(metric):
    metric.flush()
    assert os.path.exists(metric.fn)
    os.remove(metric.fn)
