"""
used for logging
"""

import os
from collections import namedtuple
from datetime import datetime
from typing import Any, Callable

from ..utils.dates import curent_date
from ..utils.screen import ScreenStr

loginfo = namedtuple('loginfo', ['string', 'prefix_len'])


class Logger:
    VERBOSE = 0
    V_DEBUG = -1
    V_INFO = 0
    V_WARN = 1
    V_ERROR = 2
    V_FATAL = 3
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        if Logger._instance is not None:
            return Logger._instance
        return super().__new__(cls)

    def __init__(self, adddate=True, datefmt: str = '%y-%m-%d %H:%M:%S', sep: str = " | ", use_stdout: bool = True):
        if Logger._instance is not None:
            return

        self.adddate = adddate
        self.datefmt = datefmt
        self.out_channel = []
        self.pipe_key = set()
        self.sep = sep
        self.return_str = ""
        self.listener = []
        self.use_stdout = use_stdout
        Logger._instance = self

    def _format(self, *values, inline=False, fix=0, raw=False, append=False):
        """"""
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
            logstr = self.sep.join((left, right))

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

    def inline(self, *values, fix=0, append=False):
        """Log a message with severity 'INFO' inline"""
        logstr, fix = self._format(*values, inline=True, fix=fix, append=append)
        self.handle(logstr, fix=fix)

    def info(self, *values):
        """Log a message with severity 'INFO'"""
        logstr, fix = self._format(*values, inline=False)
        self.handle(logstr, level=Logger.V_INFO, fix=fix)

    def raw(self, *values, inline=False, fix=0, level=0, append=False):
        """Log a message with severity 'INFO' withou datetime prefix"""
        logstr, fix = self._format(*values, inline=inline, fix=fix, raw=True, append=append)
        self.handle(logstr, level=level)

    def debug(self, *values):
        """Log a message with severity 'DEBUG'"""
        # TODO 添加红色前景
        logstr, fix = self._format("DEBUG", *values, inline=False)
        self.handle(logstr, level=Logger.V_DEBUG, fix=fix)

    def warn(self, *values):
        """Log a message with severity 'WARN'"""
        # TODO 添加黄色前景
        logstr, fix = self._format("WARN", *values, inline=False)
        self.handle(logstr, level=Logger.V_WARN, fix=fix)

    def error(self, *values):
        """Log a message with severity 'ERROR'"""
        # TODO 添加黄色背景
        logstr, fix = self._format("ERROR", *values, inline=False)
        self.handle(logstr, level=Logger.V_ERROR, fix=fix)

    def fatal(self, *values):
        """Log a message with severity 'FATAL'"""
        # TODO 添加红色背景
        logstr, fix = self._format("FATAL", *values, inline=False)
        self.handle(logstr, level=Logger.V_FATAL, fix=fix)

    def newline(self):
        """"""
        self.handle("")

    def handle(self, logstr, end="", level=0, **kwargs):
        """handle log stinrg"""
        for listener in self.listener:
            listener(logstr, end, level)

        if level < Logger.VERBOSE:
            return

        if logstr.startswith("\r"):
            fix = kwargs.get("fix", 0)
            self.return_str = logstr
            self.print(ScreenStr(logstr, leftoffset=fix), end=end)
        else:
            if len(self.return_str) != 0:
                self.print(self.return_str, end="\n")
            self.print(logstr, end="")

            for i in self.out_channel:
                with open(i, "a", encoding="utf-8") as w:
                    if len(self.return_str) != 0:
                        w.write("{}\n".format(self.return_str.strip()))
                    w.write(logstr)

            self.return_str = ""

    def print(self, *args, end='\n'):
        """built-in print function"""
        if self.use_stdout:
            print(*args, end=end, flush=True)

    def toggle_stdout(self, val:bool):
        """"""
        self.use_stdout = val

    def add_log_dir(self, dir):
        """add a file output pipeline"""
        if dir in self.pipe_key:
            self.info("Add pipe {}, but already exists".format(dir))
            return None

        os.makedirs(dir, exist_ok=True)

        i = 0
        cur_date = curent_date(fmt="%y%m%d%H%M%S")
        fni = os.path.join(dir, "l.{}.{}.log".format(cur_date, i))
        while os.path.exists(fni):
            i += 1
            fni = os.path.join(dir, "l.{}.{}.log".format(cur_date, i))

        self.print("add output channel on {}".format(fni))
        self.out_channel.append(fni)
        self.pipe_key.add(dir)
        return fni

    def add_log_listener(self, func: Callable[[str, str, int], Any]):
        """add a log event handler"""
        self.listener.append(func)

    @staticmethod
    def set_verbose(verbose=0):
        """"""
        Logger.VERBOSE = verbose
