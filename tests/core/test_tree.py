"""

"""

from lumo.core.tree import tree, Forest


def test_tree():
    tre = tree()
    tre[1][2][3][4] = 5
    assert tre[1][2][3][4] == 5
    assert isinstance(tre[1][2][3], tree), "should be {}".format(type(tree))
    assert isinstance(tre[1][1][0], tree), "should be {}".format(type(tree))


def test_forest():
    dag = Forest()

    dag.add_head('1', 1)
    dag.add_link('1', '2', 2)
    dag.add_tail('2', '3')
    dag.add_tail('1', '4', 13)
    for i in dag:
        print(i)
