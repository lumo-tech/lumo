"""
Methods about file/date/format.
"""
import hashlib
import os
from datetime import datetime

from lumo.utils.fmt import strftime


def date_from_str(value, fmt='%y-%m-%d-%H%M%S') -> datetime:
    """convert from date formatted string to datetime object"""
    return datetime.strptime(value, fmt)


def file_atime2date(file, fmt='%y%m%d-%H%M%S') -> str:
    """
    get atime of a file and convert in from timestamp to formatted string
    atime : time of last access
    """
    return strftime(fmt, datetime.fromtimestamp(os.path.getatime(file)))


def file_mtime2date(file, fmt='%y%m%d-%H%M%S') -> str:
    """
    get mtime of a file and convert in from timestamp to formatted string
    mtime : time of last modification (write)
    """
    return strftime(fmt, datetime.fromtimestamp(os.path.getmtime(file)))


def file_ctime2date(file, fmt='%y%m%d-%H%M%S') -> str:
    """
    get ctime of a file and convert in from timestamp to formatted string
    ctime : time of last status change
    """
    return strftime(fmt, datetime.fromtimestamp(os.path.getctime(file)))


def timehash():
    import time
    time_ns = str(time.time())
    hl = hashlib.md5()
    hl.update(time_ns.encode(encoding='utf-8'))
    return hl.hexdigest()
