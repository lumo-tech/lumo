from .config import glob
import pytz


def timezone():
    """
    Get the timezone from the global configuration, or default to 'Asia/Shanghai'.

    Returns:
        The timezone.
    """
    return pytz.timezone(glob.get('timezone', 'Asia/Shanghai'))
