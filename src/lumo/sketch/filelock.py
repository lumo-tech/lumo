import os
import warnings

CAN_USE_LOCK = True
if os.name == 'nt':
    import win32con, win32file, pywintypes

    LOCK_EX = win32con.LOCKFILE_EXCLUSIVE_LOCK
    LOCK_SH = 0  # The default value
    LOCK_NB = win32con.LOCKFILE_FAIL_IMMEDIATELY
    __overlapped = pywintypes.OVERLAPPED()


    def lock(file, flags):
        hfile = win32file._get_osfhandle(file.fileno())
        win32file.LockFileEx(hfile, flags, 0, 0xffff0000, __overlapped)


    def unlock(file):
        hfile = win32file._get_osfhandle(file.fileno())
        win32file.UnlockFileEx(hfile, 0, 0xffff0000, __overlapped)
elif os.name == 'posix':
    import fcntl


    def lock(file, flags=fcntl.LOCK_EX):
        fcntl.flock(file.fileno(), flags)


    def unlock(file):
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
else:
    CAN_USE_LOCK = False

    warnings.warn(f'You are in an unknown platform {os.name}, filelock may cant be used.')


    def lock(file, flat):
        raise NotImplementedError(f'an UNKNOWN platform {os.name}')


    def unlock(file):
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
