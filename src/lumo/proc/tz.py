from .config import glob
import pytz


def timezone():
    return pytz.timezone(glob.get('timezone'), 'Asia/Shanghai')
