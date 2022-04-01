"""
used for logging
"""

import logging
import os
import sys
from collections import namedtuple
from datetime import datetime
from typing import Any, Callable

from lumo.proc.dist import is_dist, local_rank
from .fmt import strftime
from .screen import ScreenStr

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

loginfo = namedtuple('loginfo', ['string', 'prefix_len'])

logger = None

VVV_DEBUG = -10
VV_DEBUG = 0
V_DEBUG = 10
V_INFO = 20
V_WARN = 30
V_ERROR = 40
V_FATAL = 50


def _get_print_func():
    try:
        from rich import print
    except ImportError:
        pass
    return print


def get_global_logger():
    global logger
    if logger is None:
        logger = Logger()
    return logger


def set_global_logger(logger_):
    global logger
    logger = logger_
    return logger


def process_str():
    if is_dist():
        return f'[{local_rank()}]'
    return ''


class Logger:
    VERBOSE = 20
    VVV_DEBUG = VVV_DEBUG
    VV_DEBUG = VV_DEBUG
    V_DEBUG = V_DEBUG
    V_INFO = V_INFO
    V_WARN = V_WARN
    V_ERROR = V_ERROR
    V_FATAL = V_FATAL
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        if Logger._instance is not None:
            return Logger._instance
        return super().__new__(cls)

    def __init__(self, adddate=True, datefmt: str = '%y-%m-%d %H:%M:%S', sep: str = " | ",
                 use_stdout: bool = True,
                 try_rich=False):
        if Logger._instance is not None:
            return

        self.adddate = adddate
        self.datefmt = process_str() + datefmt
        self.out_channel = []
        self.pipe_key = set()
        self.sep = sep
        self.return_str = ""
        self.listener = []
        self.use_stdout = use_stdout
        self._print_func = print
        self._try_rich = try_rich
        Logger._instance = self

    def _format(self, *values, inline=False, fix=0, raw=False, append=False, level=V_INFO):
        """"""
        if level < Logger.VERBOSE:
            return None, None

        if self.adddate and not raw:
            cur_date = datetime.now().strftime(self.datefmt)
        else:
            cur_date = ""

        values = ["{}".format(str(i)) for i in values]
        values = [i for i in values if len(i.strip()) != 0]

        if len(cur_date) == 0:
            space = [*["{}".format(str(i)) for i in values]]
        else:
            space = [cur_date, *values]

        if fix >= 0:
            left, right = self.sep.join(space[:fix + 1]), self.sep.join(space[fix + 1:])
            fix = len(left) + len(self.sep)
            logstr = left
            if len(right) > 0:
                logstr = self.sep.join((logstr, right))

            if inline:
                if append:
                    return "{}".format(logstr), fix
                else:
                    return "\r{}".format(logstr), fix
            else:
                return "{}\n".format(logstr), fix

        space = self.sep.join(space)

        if inline:
            return loginfo("\r{}".format(space), 0)
        else:
            return loginfo("{}\n".format(space), 0)

    def _handle(self, logstr, end="", level=10, **kwargs):
        """handle log stinrg"""
        for listener in self.listener:
            listener(logstr, end, level)

        if level < Logger.VERBOSE:
            return
        file = sys.stdout
        if level > Logger.V_INFO:
            file = sys.stderr

        if logstr.startswith("\r"):
            fix = kwargs.get("fix", 0)
            self.return_str = logstr
            self.print(ScreenStr(logstr, leftoffset=fix), end=end, file=file)
        else:
            if len(self.return_str) != 0 and not self._try_rich:
                self.print_rich(self.return_str, end="\n", file=file)
            self.print_rich(logstr, end="", file=file)

            for i in self.out_channel:
                with open(i, "a", encoding="utf-8") as w:
                    if len(self.return_str) != 0:
                        w.write("{}\n".format(self.return_str.strip()))
                    w.write(logstr)

            self.return_str = ""

    def inline(self, *values, fix=0, append=False):
        """Log a message with severity 'INFO' inline"""
        logstr, fix = self._format(*values, inline=True, fix=fix, append=append)
        if fix is None:
            return
        self._handle(logstr, level=V_INFO, fix=fix)

    def info(self, *values):
        """Log a message with severity 'INFO'"""
        logstr, fix = self._format(*values, inline=False)
        if fix is None:
            return
        self._handle(logstr, level=V_INFO, fix=fix)

    def raw(self, *values, inline=False, fix=0, level=20, append=False):
        """Log a message with severity 'INFO' withou datetime prefix"""
        logstr, fix = self._format(*values, inline=inline, fix=fix, raw=True, append=append)
        if fix is None:
            return
        self._handle(logstr, level=level)

    def debug(self, *values):
        """Log a message with severity 'DEBUG'"""
        logstr, fix = self._format("DEBUG", *values, inline=False)
        if fix is None:
            return
        self._handle(logstr, level=V_DEBUG, fix=fix)

    def ddebug(self, *values):
        """Log a message with severity 'DEBUG'"""
        logstr, fix = self._format("DDEBUG", *values, inline=False)
        if fix is None:
            return
        self._handle(logstr, level=VV_DEBUG, fix=fix)

    def dddebug(self, *values):
        """Log a message with severity 'DEBUG'"""
        logstr, fix = self._format("DDDEBUG", *values, inline=False)
        if fix is None:
            return
        self._handle(logstr, level=VVV_DEBUG, fix=fix)

    def warn(self, *values):
        """Log a message with severity 'WARN'"""
        logstr, fix = self._format("WARN", *values, inline=False)
        if fix is None:
            return
        self._handle(logstr, level=V_WARN, fix=fix)

    def error(self, *values):
        """Log a message with severity 'ERROR'"""
        logstr, fix = self._format("ERROR", *values, inline=False)
        if fix is None:
            return
        self._handle(logstr, level=V_ERROR, fix=fix)

    def fatal(self, *values):
        """Log a message with severity 'FATAL'"""
        logstr, fix = self._format("FATAL", *values, inline=False)
        if fix is None:
            return
        self._handle(logstr, level=V_FATAL, fix=fix)

    def newline(self):
        """break line"""
        self._handle("", level=V_INFO)

    def print(self, *args, end='\n', file=sys.stdout):
        """built-in print function"""
        if self.use_stdout:
            self._print_func(*args, end=end, flush=True, file=file)

    def print_rich(self, *args, end='\n', file=sys.stdout):
        if self.use_stdout:
            if self._try_rich:
                print = _get_print_func()
                print(*args, end=end, flush=True, file=file)
            else:
                self._print_func(*args, end=end, flush=True, file=file)

    def toggle_stdout(self, val: bool = None):
        """False will stop write on stdout"""
        if val is None:
            val = not self.use_stdout
        self.use_stdout = val

    def add_log_dir(self, dir, fn=None):
        """add a file output pipeline"""
        if fn is None:
            fn = ''
        else:
            fn = f'{fn}.'

        if dir in self.pipe_key:
            self.info("Add pipe {}, but already exists".format(dir))
            return None

        os.makedirs(dir, exist_ok=True)

        i = 0
        cur_date = strftime(fmt="%y%m%d%H%M")

        fmt_str = "l.{fn}{i}.{cur_date}.log"

        def _get_fn():
            return fmt_str.format(fn=fn, i=i, cur_date=cur_date)

        fni = os.path.join(dir, _get_fn())
        while os.path.exists(fni):
            i += 1
            fni = os.path.join(dir, _get_fn())

        self.print("add output channel on {}".format(fni))
        self.out_channel.append(fni)
        self.pipe_key.add(dir)
        return fni

    def add_log_listener(self, func: Callable[[str, str, int], Any]):
        """add a log event handler"""
        self.listener.append(func)

    def set_verbose(self, verbose=20):
        """set log verbose, default level is `INFO`"""
        Logger.VERBOSE = verbose
        logging.basicConfig(format='%(levelname)s:%(message)s', level=verbose)
