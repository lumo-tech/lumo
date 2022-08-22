import time

import os
import random

from lumo.utils.exithook import wrap_before


class Lock:
    def __init__(self, name, sleep=1):
        from lumo.proc.path import cache_dir
        self.file = os.path.join(cache_dir(), f"LUMO_LOCK_{name}")
        self.sleep = sleep
        wrap_before(self.clear)

    def clear(self, *_, **__):
        self.release()

    def abtain(self):
        mulp = 1
        while True:
            mulp += 1
            if mulp > 10:
                raise TimeoutError(f'Can not abtain resource of {self.file}')

            while os.path.exists(self.file):
                time.sleep(random.randint(mulp, mulp ** 2))
                mulp += 1
                if mulp > 10:
                    raise TimeoutError(f'Can not abtain resource of {self.file}')

            while True:
                flag = f'{os.getpid()}'
                with open(self.file, 'w') as w:
                    w.write(flag)

                if os.path.exists(self.file):
                    with open(self.file, 'r') as r:
                        lock_flag = r.read()
                        if flag == lock_flag:
                            return True
                        mulp += 1
                        time.sleep(random.randint(mulp, mulp ** 2))
                    if mulp > 10:
                        raise TimeoutError(f'Can not abtain resource of {self.file}')

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
