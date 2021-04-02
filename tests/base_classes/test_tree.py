"""

"""

from lumo.base_classes.tree import tree


def test_tree():
    tre = tree()
    tre[1][2][3][4] = 5
    assert tre[1][2][3][4] == 5
    assert isinstance(tre[1][2][3], tree), "should be {}".format(type(tree))
    assert isinstance(tre[1][1][0], tree), "should be {}".format(type(tree))
