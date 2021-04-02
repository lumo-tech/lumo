def test_llist():
    from lumo.base_classes import llist
    import numpy as np

    k = llist([1,2,3,4])
    assert k[np.array([0,1])] == [1,2]
    assert k[np.array(1)] == 2

