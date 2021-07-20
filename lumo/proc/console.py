import shutil
import sys
import os


def get_consolo_width():
    return shutil.get_terminal_size().columns - 1  # -1 for windows consolo


def support_multiline():
    if "jupyter_core" in sys.modules or shutil.get_terminal_size((0, 0)).columns == 0 or "PYCHARM_HOSTED" in os.environ:
        return True
    else:
        return False
