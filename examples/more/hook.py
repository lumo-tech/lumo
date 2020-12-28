"""

"""
import traceback
import atexit
import sys

class ExitHooks(object):
    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        with open("1","w") as f:
            f.write(str(code))
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, tb,*args):
        self.exception = exc
        # print(traceback.format_exception(exc_type,exc,tb))
        # print(traceback.format_exception_only(exc_type,exc))
        # traceback.print_exception(exc_type,exc,tb)

hooks = ExitHooks()
hooks.hook()

raise Exception("asdasdasd")