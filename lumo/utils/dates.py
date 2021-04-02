"""
Methods about file/date/format.
"""
import os
from datetime import datetime


def strftime(fmt='%y-%m-%d-%H%M%S', dateobj: datetime = None):
    """get current date with formatted"""
    if dateobj is not None:
        return dateobj.strftime(fmt)
    return datetime.now().strftime(fmt)


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
