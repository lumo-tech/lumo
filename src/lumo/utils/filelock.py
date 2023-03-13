from filelock import Timeout, FileLock
import os


class Lock:
    """
    A class for obtaining and releasing file-based locks using FileLock.

    Args:
        name (str): The name of the lock.

    Attributes:
        fn (str): The file path of the lock file.
        lock (FileLock): The FileLock object used for obtaining and releasing the lock.

    Example:
        lock = Lock('my_lock')
        lock.abtain()
        # critical section
        lock.release()
    """

    def __init__(self, name):
        """Initialize the lock file path and FileLock object"""
        from lumo.proc.path import cache_dir
        self.fn = os.path.join(cache_dir(), f"LUMO_LOCK_{name}")
        self.lock = FileLock(self.fn)

    def abtain(self):
        """Acquire the lock"""
        self.lock.acquire()

    def release(self):
        """Release the lock"""
        self.lock.release()
