import textwrap
from pprint import pformat
from typing import List, Union, Any


def _safe_repr(values: Any) -> str:
    return pformat(values)


def safe_param_repr(values: List[tuple], level=1) -> str:
    """

    Args:
        values:
        level:

    Returns:

    """
    res = '\n'.join([f"{k}={_safe_repr(v)},  # {anno}" for k, v, anno in values])
    return textwrap.indent(res, '    ')
