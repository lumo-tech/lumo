from itertools import zip_longest, chain, accumulate, repeat
from operator import add
from typing import Sized, Iterable
from typing import Sequence


def chunk(seq: Sequence, chunksize=10, pad_last=False, pad_item=None):
    ress = [iter(seq)] * chunksize

    for res in zip_longest(*ress, fillvalue=pad_item):
        if not pad_last:
            res = [i for i in res if i is not None]
        yield res


def window(seq: Sized, n: int, strid: int = 1, drop_last: bool = False):
    for i in range(0, len(seq), strid):
        res = seq[i:i + n]
        if drop_last and len(res) < n:
            break
        yield res


def window2(seq: Iterable, n: int, strid: int = 1, drop_last: bool = False):
    it = iter(seq)
    result = []
    step = 0
    for i, ele in enumerate(it):
        result.append(ele)
        result = result[-n:]
        if len(result) == n:
            if step % strid == 0:
                yield result
            step += 1
    if not drop_last:
        yield result


def lfilter(func, iterable):
    return list(filter(func, iterable))


def lmap(func, iterable):
    return list(map(func, iterable))


def lchain(*iterable):
    return list(chain(*iterable))


def poll(*iterables, c=None):
    iterables = [safe_cycle(i) for i in iterables]
    # TODO
    if c is not None:
        pass

    while True:
        for iterable in iterables:
            yield next(iterable)


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


def accumulate_slice(iterable):
    """
    ```
    for sl in list(accumulate_slice(range(10))):
        print(list(range(45)[sl]))

    []
    [0]
    [1, 2]
    [3, 4, 5]
    [6, 7, 8, 9]
    [10, 11, 12, 13, 14]
    [15, 16, 17, 18, 19, 20]
    [21, 22, 23, 24, 25, 26, 27]
    [28, 29, 30, 31, 32, 33, 34, 35]
    [36, 37, 38, 39, 40, 41, 42, 43, 44]

    ```
    Args:
        iterable:

    Returns:

    """
    offset, acc = repeat(iterable, 2)
    for a, b in zip(offset, accumulate(acc, add)):
        yield slice(b - a, b)


def groupby(iterable, key=None):
    """

    >>> labels = [1,1,2,2,3,3,4,4]
    >>> for cur,vals in list(groupby(labels)):
    >>>     print(cur,vals)

        1 [0, 1]
        2 [2, 3]
        3 [4, 5]
        4 [6, 7]

    Args:
        iterable:
        key:

    Returns:

    """
    if key is None:
        key = lambda a, b: a == b

    res = []
    cur = None
    for i, val in enumerate(iterable):
        if i == 0:
            cur = val
        else:
            if not key(cur, val):
                yield cur, res
                res = []
        cur = val
        res.append(i)
    yield res


def group_continuous(iterable, key=None):
    """
    Returns grouped items constrained by continuous integer in the iterable object.

    >>> list(group_continuous([[i] for i in [1,2,3,4,5,7,8,9,12,13]], key=lambda x:x[0]))
        [[[1], [2], [3], [4], [5]], [[7], [8], [9]], [[12], [13]]]


    >>> list(group_continuous([1,2,3,4,5,7,8,9,12,13]))
        [[1, 2, 3, 4, 5], [7, 8, 9], [12, 13]]
    """
    cur = -1
    offset = 0
    res = []
    for item in iterable:
        if key is None:
            i = item
        else:
            i = key(item)
        if cur == -1:
            cur = i
        else:
            offset = i - cur
            cur = i
        if offset > 1:
            yield res
            res = []
        res.append(item)
    yield res
