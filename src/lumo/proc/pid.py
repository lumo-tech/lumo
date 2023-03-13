"""
Returns information about the specified process or the current process, and computes its hash value.
"""
from psutil import Process, pid_exists
from joblib import hash
import os


def runtime_pid_obj(pid=None):
    """Returns a dictionary containing information about a process identified by the given PID.

    Args:
        pid (int, optional): The PID of the process to get information about. If None, uses the PID of the current process. Defaults to None.

    Returns:
        dict: A dictionary containing the following keys:
            - pid (int): The process ID.
            - pname (str): The name of the process.
            - pstart (float): The process creation time, in seconds since the epoch.
            - argv (list): A list of command-line arguments passed to the process.
    """
    if pid is None:
        pid = os.getpid()

    if pid_exists(pid):
        p = Process(pid)
        obj = {
            "pid": p.pid, "pname": p.name(), 'pstart': p.create_time(), 'argv': p.cmdline()
        }
        return obj

    return None


def pid_hash(pid_obj=None):
    """Computes the hash of a process object.

    Args:
        pid_obj (dict, optional): A dictionary containing information about a process. If None, uses the information about the current process. Defaults to None.

    Returns:
        str: The hash of the process object.
    """
    if pid_obj is None:
        pid_obj = runtime_pid_obj()
    return hash(pid_obj)
