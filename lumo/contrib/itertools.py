from itertools import zip_longest
from typing import Sequence


def chunk(seq: Sequence, chunksize=10, pad_last=False, pad_item=None):
    ress = [iter(seq)] * chunksize

    for res in zip_longest(*ress, fillvalue=pad_item):
        if not pad_last:
            res = [i for i in res if i is not None]
        yield res
