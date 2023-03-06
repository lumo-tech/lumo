"""
Methods about screen single line outputs.
"""
import os
import shutil
import sys
import time

from tqdm import tqdm


def get_consolo_width():
    """Returns the width of the current console window"""
    return shutil.get_terminal_size().columns - 1  # -1 for windows consolo


def support_multiline():
    """
    Checks if the current environment supports multiline output in line.
    Notes:
        This function checks if the current environment supports multiline output.
        It returns True if any of the following conditions are met:

        - The `jupyter_core` module is available (implying that the code is being run in a Jupyter notebook or JupyterLab).
        - The width of the console is reported as 0 by `shutil.get_terminal_size()`, which can occur in some non-standard environments or configurations.
        - The `PYCHARM_HOSTED` environment variable is set, which indicates that the code is being run in PyCharm's integrated console.

        If none of these conditions are met, the function returns False.
    """
    if "jupyter_core" in sys.modules or shutil.get_terminal_size((0, 0)).columns == 0 or "PYCHARM_HOSTED" in os.environ:
        return True
    else:
        return False


def _is_jupyter() -> bool:  # pragma: no cover
    """Check if we're running in a Jupyter notebook."""
    try:
        get_ipython  # type: ignore
    except NameError:
        get_ipython = lambda: ()
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
    A class representing a string that can be displayed on a console screen.

    Attributes:
        content (str): The string content to be displayed on the screen.
        leftoffset (int): The number of characters to shift the content to the left.

    Notes:
        A ScreenStr starting with '\r' will not overflow, and any string longer than the screen width will be cut off.
        If the console supports multiline output (like PyCharm or Jupyter notebook), all the string will be represented.
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
        """Initializes a new instance of the ScreenStr class."""
        self.content = content
        ScreenStr.left = leftoffset

    def __repr__(self) -> str:
        """Returns the string representation of the ScreenStr object."""
        if ScreenStr.multi_mode:
            return self.content
        return self._screen_str()

    def __len__(self) -> int:
        """Returns the length of the string content."""
        txt = self.content.encode("gbk", errors='ignore')
        return len(txt)

    def tostr(self):
        """Returns the string content."""
        return self.content

    @classmethod
    def set_speed(cls, dt: float = 0.05):
        """Sets the speed of the text scrolling animation."""
        cls.dt = dt

    @classmethod
    def deltatime(cls):
        """Calculates the time elapsed since the last update."""
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
        """Calculates the offset for scrolling the text."""

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

    def _decode_sub(self, txt, left, right):
        """Decodes a part of a byte string to a Unicode string."""
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
        """Returns the width of the console."""
        width = get_consolo_width()
        return width

    @staticmethod
    def split(txt, len):
        """Splits a string into two parts."""
        try:
            return txt[:len], txt[len:]
        except:
            try:
                return txt[:len + 1], txt[len + 1:]
            except:
                return txt[:len - 1], txt[len - 1:]

    def _screen_str(self, margin="..."):
        """Returns the string content formatted for display on the screen."""
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
    """
    A subclass of `tqdm` that formats progress bar updates as a single line.

    This subclass provides two additional methods:

        - `full_str`: Returns a formatted string representing the full progress bar,
         including the progress bar itself and any additional information (such as elapsed time or estimated remaining time).

    """

    def full_str(self):
        """
        Returns a formatted string representing the full progress bar, including the progress bar itself and any additional information (such as elapsed time or estimated remaining time).

        Args:
            None

        Returns:
            str: A formatted string representing the full progress bar.
        """
        return self.format_meter(**self.format_dict)

    def __str__(self):
        """
        Overrides the `__str__` method of the `tqdm` class to display the progress bar as a single-line string.

        Args:
            None

        Returns:
            str: A single-line string representation of the progress bar.
        """
        return ScreenStr(self.full_str())._screen_str()
