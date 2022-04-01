from lumo.contrib.optim import lr_scheduler
from lumo import attr


def test_scheduler_is_attr():
    a = attr()
    a.sche = lr_scheduler.CosScheduler(start=0, end=1, left=0, right=1)
    assert a.sche.get(0) == a.sche(0) == 0

    b = attr.from_dict(a.jsonify())
    assert b.sche.__class__.__name__ == lr_scheduler.CosScheduler.__name__
    assert b.sche.get(0) == b.sche(0) == 0
    assert a.hash() == b.hash()
