from itertools import zip_longest, chain
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
