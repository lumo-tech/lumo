"""

"""

from lumo.base_classes.metaclasses import Merge

def test_merge():
    class A(metaclass=Merge):
        _item = {1:2,3:4}

    class B(A):
        _item = {5:6,7:8}

    b = B()
    assert 1 in b._item and 3 in b._item and 5 in b._item and 7 in b._item
    assert 1 in B._item and 3 in B._item and 5 in B._item and 7 in B._item

