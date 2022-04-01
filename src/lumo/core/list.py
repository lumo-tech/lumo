"""

"""
import numbers
from typing import Any, Iterable

import numpy as np
import torch


class llist(list):
    """
    slice as you like

    Examples:
    >>> res = llist([1,2,3,4])
    ... print(res[0])
    ... print(res[0:3])
    ... print(res[0,2,1])

    >>> idx = torch.randperm(3)
    ... print(res[idx])

    >>> idx = torch.randint(0,3,[3,4])
    ... print(res[idx])

    >>> idx = np.array([1,2,-3])
    ... print(res[idx])

    >>> idx = np.array(2)
    ... print(res[idx])
    """

    def __getitem__(self, i: [int, slice, Iterable]) -> Any:
        if isinstance(i, (slice, numbers.Integral)):
            # numpy.int64 is not an instance of built-in type int
            try:
                res = super().__getitem__(i)
            except IndexError:
                raise IndexError('list index out of range, got {}, but max is {}'.format(i, len(self) - 1))

            if isinstance(res, list):
                return llist(res)
            else:
                return res
        elif isinstance(i, (Iterable)):
            if isinstance(i, torch.Tensor):
                if len(i.shape) == 0:
                    i = i.item()
                    return self.__getitem__(i)
                else:
                    i = i.tolist()
            if isinstance(i, np.ndarray):
                if i.dtype == np.bool:
                    i = np.where(i)[0]

                if len(i.shape) == 0:
                    i = i.tolist()
                    return self.__getitem__(i)
                else:
                    i = i.tolist()

            return llist(self.__getitem__(id) for id in i)
