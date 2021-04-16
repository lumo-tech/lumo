try:
    import regex as re
except:
    import re

from .device import to_device, to_device_enumrate, construct_device_args_kwargs
from .keys import K
from . import safe_io
