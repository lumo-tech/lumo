"""


    提供一些提供了奇葩 feature 的类
"""
from typing import Any


class NoneItem:
    """
    Can be seen as identity element or zero element
    """

    @staticmethod
    def clone(x):
        pass

    def __eq__(self, o: object) -> bool:
        return o is None

    def __add__(self, other):
        return 0 + other

    def __mul__(self, other):
        return 1 * other

    def __sub__(self, other):
        return 0 - other

    def __truediv__(self, other):
        return 1 / other

    def __floordiv__(self, other):
        return 1 / other

    def __repr__(self):
        return "NoneItem()"

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __gt__(self, other):
        return True

    def __cmp__(self, other):
        return True

    def __ne__(self, other):
        return other is not None

    def __int__(self):
        return 0

    def __float__(self):
        return 0


class AvgItem:
    """
    用于保存累积均值的类
    avg = AvgItem()
    avg += 1 # avg.update(1)
    avg += 2
    avg += 3

    avg.item = 3 #(last item)
    avg.avg = 2 #(average item)
    avg.sum = 6
    """

    def __init__(self, weight=1) -> None:
        super().__init__()
        self._sum = 0
        self._weight = weight
        self._count = 0
        self._item = 0

    def __add__(self, other):
        self.update(other)
        return self

    def update(self, other, weight=1):
        self._sum += other * self._weight
        self._count += self._weight
        self._item = other

    @property
    def item(self):
        return self._item

    @property
    def avg(self):
        if self._count == 0:
            return 0
        return self._sum / self._count

    def __repr__(self) -> str:
        return str("{}({})".format(self._item, self.avg))

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __getattr__(self, item):
        return getattr(self._item, item)

    def __format__(self, format_spec):
        return "{{:{}}}({{:{}}})".format(format_spec, format_spec).format(self._item, self.avg)

    def __getitem__(self, item):
        return self._item[item]


