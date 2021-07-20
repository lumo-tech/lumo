"""
Methods about screen single line outputs.
"""
import sys
import time
import collections
from lumo.proc.console import get_consolo_width, support_multiline
import numpy as np


class ScreenStr:
    """
    A ScreenStr start with '\r' won't overflow, any string outside the screen width will be cut.

    Notes:
    If output consolo support multiline(like pycharm or jupyter notebook) return, all string will be represented.
    """
    t = 0
    dt = 0.7
    last = 0
    left = 0
    max_wait = 1.
    wait = 0
    wait_toggle = False

    debug = False
    last_width = 0
    multi_mode = support_multiline()

    def __init__(self, content="", leftoffset=0) -> None:
        self.content = content
        ScreenStr.left = leftoffset

    def __repr__(self) -> str:
        if ScreenStr.multi_mode:
            return self.content
        return self._screen_str()

    def tostr(self):
        return self.content

    @classmethod
    def set_speed(cls, dt: float = 0.05):
        cls.dt = dt

    @classmethod
    def deltatime(cls):
        if cls.last == 0:
            cls.last = time.time()
            return 0
        else:
            end = time.time()
            res = end - cls.last
            cls.last = end
            return res

    @classmethod
    def cacu_offset_(cls, out_width):

        delta = cls.deltatime()
        cls.t += delta * cls.dt

        # pi = 2*math.pi
        t = cls.t
        # k = 2 * out_width / pi
        k = 10
        pi = 2 * out_width / k
        offset = round(k * (t % pi) * ((t % pi) < pi / 2) + (-k * (t % pi) + 2 * out_width) * ((t % pi) > pi / 2))

        # offset = math.floor(out_width * (math.cos(ScreenStr.t + math.pi) + 1) / 2)
        # print(offset)
        return offset

    a = 1

    def __len__(self) -> int:
        txt = self.content.encode("gbk", errors='ignore')
        return len(txt)

    def _decode_sub(self, txt, left, right):
        try:
            txt = txt[left:right].decode("gbk", errors='ignore')
        except:
            try:
                txt = txt[left:right - 1].decode("gbk", errors='ignore')
            except:
                try:
                    txt = txt[left + 1:right].decode("gbk", errors='ignore')
                except:
                    txt = txt[left + 1:right - 1].decode("gbk", errors='ignore')

        return txt

    @staticmethod
    def consolo_width():
        width = get_consolo_width()
        return width

    @staticmethod
    def split(txt, len):
        try:
            return txt[:len], txt[len:]
        except:
            try:
                return txt[:len + 1], txt[len + 1:]
            except:
                return txt[:len - 1], txt[len - 1:]

    def _screen_str(self, margin="..."):
        width = self.consolo_width()

        txt = self.content.encode("gbk", errors='ignore').strip()
        textlen = len(txt)

        if textlen <= width:
            return self.content

        left, right = self.split(txt, self.left)
        if len(left) >= width:
            return left[:width]

        offset = 0  # TODO

        offright = width - len(left) + offset - len(margin)

        left = left.decode("gbk", errors='ignore')
        right = self._decode_sub(right, offset, offright)

        head = "\r" if self.content.startswith("\r") else ""
        tail = "\n" if self.content.endswith("\n") else ""

        txt = "{}{}{}{}".format(head, left, right, tail)
        return txt + margin


class Progbar(object):
    """从 keras 中扒下来的进度条显示类

    Displays a progress bar.
    Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """
        Updates the progress bar.

        Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        strs = []
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            strs.append(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            strs.append(info)

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                strs.append(info)

        self._last_update = now

        return "".join(strs)

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
