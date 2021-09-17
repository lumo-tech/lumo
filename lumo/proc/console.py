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
