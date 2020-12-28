"""
Methods about iterator
"""
from collections.abc import Iterator


class safe_cycle():
    class _cycle_none():
        pass

    def __init__(self, iter_item):
        self.item = iter_item
        self.iter = None

    def __next__(self):
        if self.iter is None:
            self.iter = iter(self.item)
            cur = next(self.iter, safe_cycle._cycle_none())
            if isinstance(cur, safe_cycle._cycle_none):
                raise StopIteration('item has no element.')
            return cur
        else:
            cur = next(self.iter, safe_cycle._cycle_none())
            if not isinstance(cur, safe_cycle._cycle_none):
                return cur
            else:
                self.iter = iter(self.item)
                return next(self)

    def __iter__(self):
        while True:
            yield next(self)


def deep_chain(item):
    """flatten iterator"""
    if isinstance(item, Iterator):
        for i in item:
            if isinstance(i, Iterator):
                for ii in deep_chain(i):
                    yield ii
            else:
                yield i
    else:
        yield item


def is_same_type(items, ty=None):
    for item in items:
        if ty is None:
            ty = type(item)
        else:
            if type(item) != ty:
                return False
    return True
