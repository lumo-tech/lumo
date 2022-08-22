import time

import os
import random
from joblib import hash
import mmap
from lumo.utils.exithook import wrap_before


class Lock:
    def __init__(self, name, sleep=1, group=None):
        from lumo.proc.path import cache_dir
        if group is None:
            group = ""

        self.size = 10000
        self.flag = f'{os.getpid()}'
        self.flagsize = len(self.flag)
        self.pos = int(hash(name), 16) % (self.size - len(self.flag))
        self.file = os.path.join(cache_dir(), f'LUMO_LOCK_V3_{group}')
        if not os.path.exists(self.file):
            with open(self.file, 'w') as w:
                w.write('0' * self.size)

        self.sleep = sleep
        self.fhdl = None
        wrap_before(self.clear)

    def clear(self, *_, **__):
        self.release()

    def abtain(self):
        mulp = 0
        self.fhdl = r = open(self.file, 'r+b')
        mm = mmap.mmap(r.fileno(), 0, access=mmap.ACCESS_WRITE)
        flag = f'{os.getpid()}'
        while True:
            mulp += 1
            if mulp > 10:
                raise TimeoutError(f'Can not abtain resource of {self.file}')

            if mm[self.pos:self.pos + len(flag)].decode() != '0' * len(flag):
                time.sleep(random.randint(mulp, mulp ** 2))
                continue

            mm[self.pos:self.pos + len(flag)] = flag.encode()
            mm.flush()

            if mm[self.pos:self.pos + len(flag)].decode() != flag:
                mm.close()
                time.sleep(random.randint(mulp, mulp ** 2))
                continue

            return True

    def release(self):
        if self.fhdl is None:
            return

        r = self.fhdl
        mm = mmap.mmap(r.fileno(), 0, access=mmap.ACCESS_WRITE)
        flag = b'0' * self.flagsize
        mm[self.pos:self.pos + len(flag)] = flag
        mm.flush()
        r.close()
        self.fhdl = None
        return True

    def __enter__(self):
        self.abtain()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
