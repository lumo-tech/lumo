"""
Methods about time.
"""
import pprint
import time
import warnings
from collections import OrderedDict

from .fmt import strftime


def format_second(sec: int) -> str:
    """convert seconds from int to string"""
    sec, ms = divmod(sec, 1)
    if sec > 60:
        min, sec = divmod(sec, 60)
        if min > 60:
            hour, min = divmod(min, 60)
            fmt = "{}h{}m{}s".format(hour, min, int(sec))
        else:
            fmt = "{}m{}s".format(min, int(sec))
    else:
        fmt = "{}s".format(int(sec))
    return fmt


class Timer:
    """
    A class for timing the time cost in each part.

    A global object is contained in lumo:

    ```python
    from lumo.utils import timeit
    timeit.start()
    timeit.mart("load")
    timeit.end("load")
    print(timeit.meter())
    ```

    """

    def __init__(self):
        self.last_update = None
        self.ends = False
        self.times = OrderedDict()

    def offset(self):
        now = time.time()

        if self.last_update is None:
            offset = 0
        else:
            offset = now - self.last_update

        self.last_update = now
        return offset, now

    def clear(self):
        self.last_update = None
        self.ends = False
        self.times.clear()

    def start(self):
        self.clear()
        self.mark("start", True)

    def mark(self, key, add_now=False):
        if self.ends:
            warnings.warn("called end method, please use start to restart timeit")
            return
        key = str(key)
        offset, now = self.offset()

        self.times.setdefault("use", 0)
        self.times["use"] += offset

        if add_now:
            self.times[key] = strftime("%H:%M:%S")
        else:
            self.times.setdefault(key, 0)
            self.times[key] += offset

    def end(self):
        self.mark("end", True)
        self.ends = True

    def meter(self, ratio=True):
        from lumo import Meter

        meter = Meter()
        for key, offset in self.times.items():
            if ratio:
                if isinstance(offset, str):
                    continue
                meter[key] = offset / self.times['use']
            else:
                meter[key] = offset

        return meter

    def __str__(self):
        return pprint.pformat(self.times)

    def __getitem__(self, item):
        return self.times[item]

    def __getattr__(self, item):
        return self.times[item]


timer = Timer()
