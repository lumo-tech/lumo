import textwrap
from pprint import pformat
from typing import List, Union, Any


def _safe_repr(values: Any) -> str:
    return pformat(values)


def _padding_mod(st: str, offset=7, mod=4):
    """
    123 \\
    1   \\
    12312341    \\
    1231
    Args:
        strs:
        mod:

    Returns:

    """
    size = len(st)
    if size < offset:
        return st.ljust(offset, ' ')

    mnum = mod - len(st) % mod
    # if mnum == 0:
    #     mnum = mod
    return st.ljust(size + mnum, ' ')


def safe_param_repr(values: List[tuple], level=1) -> str:
    """

    Args:
        values:
        level:

    Returns:

    """
    res = [(f"{k}={_safe_repr(v)},", anno) for k, v, anno in values]

    # res = textwrap.fill('\n'.join(res))
    res = '\n'.join([_padding_mod(i, offset=16, mod=4) + f'  # {anno}' for i, anno in res])

    return textwrap.indent(res, '    ')
