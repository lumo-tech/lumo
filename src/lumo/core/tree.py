"""

"""
import queue
from collections import defaultdict


class tree(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def walk(self):
        for k, v in self.items():
            yield k, v
            if isinstance(v, tree):
                for kk, vv in v.walk():
                    yield f'{k}/{kk}', vv


class Node:
    HEAD = 0
    MID = 1
    TAIL = 2

    def __init__(self):
        self.value = None
        self.link = []
        self.stage = None

    @property
    def is_head(self):
        return self.stage == self.HEAD

    @property
    def is_mid(self):
        return self.stage == self.MID

    @property
    def is_tail(self):
        return self.stage == self.TAIL

    def set_stage(self, stage):
        self.stage = stage
        return self

    def set_value(self, val):
        self.value = val
        return self

    def add_link(self, y):
        self.link.append(y)

    def __repr__(self):
        return f'Node({self.stage} ,{len(self.link)}, {self.value})'


class Forest:
    def __init__(self):
        self.dic = defaultdict(Node)
        self.order = []
        self.tail = set()

    def add_head(self, x, val=None):
        self.dic[x].set_value(val).set_stage(Node.HEAD)
        self.order.append(x)
        return self

    def check_node_type(self, x):
        return x in self.dic

    def add_link(self, x, y, y_val=None):
        assert x in self.dic, f'x must already existed in graph, has {self.order}, got {x}'
        assert y not in self.dic, f'y must be a new node in graph, has {self.order}, got {y}'
        self.dic[x].add_link(y)
        self.dic[y].set_value(y_val).set_stage(Node.MID)
        self.order.append(y)
        return self

    def add_tail(self, x, y, y_val=None):
        assert x in self.dic, f'x must already existed in graph, has {self.order}, got {x}'
        assert y not in self.dic, f'y must be a new node in graph, has {self.order}, got {y}'
        self.dic[x].add_link(y)
        self.dic[y].set_value(y_val).set_stage(Node.TAIL)
        self.order.append(y)
        self.tail.add(y)
        return self

    def __iter__(self):
        stack = []
        mem = set()

        if len(self.order) > 0:
            stack.append(self.order[0])

        while len(stack) > 0:
            key = stack.pop(0)
            if key in mem:
                continue
            yield key, self.dic[key]

            mem.add(key)
            for key in self.dic[key].link:
                stack.append(key)


if __name__ == '__main__':
    dag = Forest()

    dag.add_head('1', 1)
    dag.add_link('1', '2', 2)
    dag.add_tail('2', '3')
    dag.add_tail('1', '4', 13)
    for i in dag:
        print(i)
