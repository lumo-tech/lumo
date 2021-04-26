try:
    import regex as re
except ImportError:
    import re

from .device import to_device, to_device_enumrate, construct_device_args_kwargs
from .keys import K
from . import safe_io
