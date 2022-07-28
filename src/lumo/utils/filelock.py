import time

from ..proc.path import cache_dir
import os
import random


class Lock:
    def __init__(self, name, sleep=1):
        self.file = os.path.join(cache_dir(), f'LUMO_LOCK_{name}')
        self.sleep = sleep

    def abtain(self):
        while True:
            mulp = 1
            while os.path.exists(self.file):
                time.sleep(self.sleep)

            flag = f'{os.getpid()}'
            with open(self.file, 'w') as w:
                w.write(flag)
            with open(self.file, 'r') as r:
                lock_flag = r.read()
                if flag == lock_flag:
                    return True
                mulp += 1
                time.sleep(random.randint(mulp, mulp ** 2))

    def release(self):
        try:
            os.remove(self.file)
        except:
            pass
        return True

    def __enter__(self):
        self.abtain()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
