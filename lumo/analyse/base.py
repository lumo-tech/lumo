from typing import List, Union, Dict, Any, Sequence
from lumo.utils.fmt import to_ndarray
from dataclasses import dataclass


@dataclass()
class Mile:
    step: int
    left = None
    right = None


def fmt(lst: Union[List, Dict[int, Any]]) -> Dict[int, float]:
    if isinstance(lst, dict):
        return {k: to_ndarray(v) for k, v in lst.items()}
    if isinstance(lst, list):
        typ = type(lst[0])
        if isinstance(typ, dict):
            lst = {int(step): to_ndarray(val) for step, val in lst}
        elif isinstance(typ, Sequence):
            lst = {int(i): to_ndarray(val) for i, val in lst}
        else:
            lst = {i: to_ndarray(val) for i, val in enumerate(lst)}
        return lst


def cmp(left: Union[List, Dict], right: Union[List, Dict]):
    left, right = fmt(left), fmt(right)

    merge = {}

    for k, v in left.items():
        inner = merge.setdefault(k, Mile(k))
        inner.left = v

    for k, v in right.items():
        inner = merge.setdefault(k, Mile(k))
        inner.right = v

    return sorted(list(merge.values()), key=lambda x: x.step)


def pop_empty(merged: List[Mile]):
    return [i for i in merged if i.left is not None and i.right is not None]
