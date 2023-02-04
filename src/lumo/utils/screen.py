"""
Methods about screen single line outputs.
"""
import os
import shutil
import sys
import time

from tqdm import tqdm


def get_consolo_width():
    return shutil.get_terminal_size().columns - 1  # -1 for windows consolo


def support_multiline():
    if "jupyter_core" in sys.modules or shutil.get_terminal_size((0, 0)).columns == 0 or "PYCHARM_HOSTED" in os.environ:
        return True
    else:
        return False


def _is_jupyter() -> bool:  # pragma: no cover
    """Check if we're running in a Jupyter notebook."""
    try:
        get_ipython  # type: ignore
    except NameError:
        return False
    ipython = get_ipython()  # type: ignore
    shell = ipython.__class__.__name__
    if "google.colab" in str(ipython.__class__) or shell == "ZMQInteractiveShell":
        return True  # Jupyter notebook or qtconsole
    elif shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)


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

        offset = 0

        offright = width - len(left) + offset - len(margin)

        left = left.decode("gbk", errors='ignore')
        right = self._decode_sub(right, offset, offright)

        head = "\r" if self.content.startswith("\r") else ""
        tail = "\n" if self.content.endswith("\n") else ""

        txt = "{}{}{}{}".format(head, left, right, tail)
        return txt + margin


class inlinetqdm(tqdm):

    def full_str(self):
        return self.format_meter(**self.format_dict)

    def __str__(self):
        return ScreenStr(self.full_str())._screen_str()
