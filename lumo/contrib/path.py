import os
from typing import Tuple, Iterable, List


def walk(top: str, depth: int = -1) -> Iterable[Tuple[str, List[str], List[str]]]:
    """same as os.path.walk but added depth argument"""
    fs = os.listdir(top)

    dirs = []
    files = []
    rfs = [os.path.join(top, i) for i in fs]
    [dirs.append(i) if os.path.isdir(j) else files.append(i) for i, j in zip(fs, rfs)]
    yield top, dirs, files
    if depth > 0:
        for dir in dirs:
            yield from walk(os.path.join(top, dir), depth - 1)