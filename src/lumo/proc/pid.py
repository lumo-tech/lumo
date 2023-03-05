from psutil import Process
import sys
from joblib import hash
import os


def runtime_pid_obj(pid=None):
    if pid is None:
        pid = os.getpid()
    p = Process(pid)
    obj = {
        "pid": p.pid, "pname": p.name(), 'pstart': p.create_time(), 'argv': p.cmdline()
    }
    return obj


def pid_hash(pid_obj=None):
    if pid_obj is None:
        pid_obj = runtime_pid_obj()
    return hash(pid_obj)
