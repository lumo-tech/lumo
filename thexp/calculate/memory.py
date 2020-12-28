"""

"""
import operator as op
import sys
from functools import reduce
from typing import Union

import numpy as np


def pin(*sizes, dtype: str = "float32", format: bool = False) -> Union[int, str]:
    """
    计算相应尺寸和类型“至少”占用的内存大小
    由于除了存放相应 size 个数的数据外，还有用于辅助的各类变量，因此说是至少

    Args:
        *sizes:
        dtype:
        format:

    Returns:
        if format:
            return 格式化后的大小
        else:
            return 整数，表示 byte 数
    Examples:
    >>> print(pin(10,50000,format=True))
    """
    # len(sizes) * 16 + 80
    one = np.array(1, dtype=dtype)

    size = sys.getsizeof(one) - 80
    mem = reduce(op.mul, sizes) * size

    if format:
        unit = "B"
        if mem > 1024:
            mem /= 1024
            unit = "KB"
            if mem > 1024:
                mem /= 1024
                unit = "MB"
                if mem > 1024:
                    mem /= 1024
                    unit = "GB"
        return "{:.4f} {}".format(mem, unit)

    return mem
