import pathlib
import sys
from typing import Union

if sys.version_info >= (3, 6):
    PathLike = Union[str, "os.PathLike[str]"]
else:
    PathLike = Union[str, pathlib.PurePath]
