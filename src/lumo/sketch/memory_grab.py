"""
A hack function to monitor GPU memory usage, then occupy its access.
"""
import functools
import os
import subprocess
import time
from functools import partial

import torch

from lumo.core.tree import tree
from lumo.utils import re

match_mem = re.compile('([0-9]+) +([0-9]+)[^|]* ([0-9]+)MiB')


class DeviceMem:
    def __init__(self):
        self.line_mem = tree()

    def _parse_device_pid_mem_pair(self, lines):
        for lid, line in enumerate(lines):
            res = re.search(match_mem, line)
            if res is not None:
                _device, _pid, _mib = [int(i) for i in res.groups()]
                self.line_mem[_device][_pid] = lid
                yield _device, _pid, _mib

    def try_parse(self, lines, pid, device):
        """ try parse mem from cached lid directly.
        Returns:
             -1 means failed.
             others means successd and its memory.

        """
        lid = self.line_mem[device][pid]
        if isinstance(lid, dict):
            return -1
        elif lid > len(lines):
            return -1
        else:
            res = re.search(match_mem, lines[lid])
            if res is None:
                return -1
            else:
                _device, _pid, _mib = [int(i) for i in res.groups()]
                if _pid == pid and _device == device:
                    return _mib
                else:
                    return -1

    def re_parse(self, lines, pid, device):
        _res = self.try_parse(lines, pid, device)
        if _res != -1:
            return _res

        for _device, _pid, _mib in self._parse_device_pid_mem_pair(lines):
            if _pid == pid and _device == device:
                return _mib

        return 0

    def _get_nvidia_smi(self):
        proc = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
        lines = proc.stdout.readlines()
        return [i.decode() for i in lines]

    def _device_equal(self, da, db):
        if isinstance(da, (int, str)):
            da = torch.device(da)
        if isinstance(db, (int, str)):
            db = torch.device(db)
        return da == db

    def get_device_release_mem(self, device):
        """ get device memory left."""
        s_pid = os.getpid()
        total = self.get_device_mem(device)
        for _device, _pid, _mib in self._parse_device_pid_mem_pair(self._get_nvidia_smi()):
            if self._device_equal(device, _device):
                total -= _mib

        return total

    def get_device_mem(self, device):
        """  returns device total memory(unit: MB)        """
        return torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)

    def get_pid_device_mem(self, pid, device):
        """
        尽可能有效率的得到进程在某设备下占用的显存（通过命令行程序调用获取）
        :param pid:
        :param device:
        :return:
        """
        if isinstance(device, torch.device):
            device = device.index
        elif isinstance(device, str):
            device = torch.device(device).index

        lines = self._get_nvidia_smi()
        mib = self.try_parse(lines, pid, device)
        if mib == -1:
            mib = self.re_parse(lines, pid, device)

        return mib


_memer = DeviceMem()
_pid = os.getpid()

if torch.cuda.is_available():
    get_pid_device_mem = partial(_memer.get_pid_device_mem, pid=_pid, device=torch.cuda.current_device())


class memory(object):
    r"""
    优雅的抢卡
    Args:
        memory: 需要占用的内存，以 MB 为单位
        device: 需要占用内存的设备
        hold:
        unit:
    Example::
        >>> import lumo
        >>> with lumo.memory(5000):
        ...   y = x * 2

        >>> @lumo.memory(1024)
        ... def doubler(x):
        ...     ...

        >>> lumo.memory(10000).start()
        ... # do something

    Why use nvidia-smi to get memory useage? see:
        https://github.com/pytorch/pytorch/issues/12873
    """

    def __init__(self, memory, device=None, hold=False) -> None:
        super().__init__()
        if device is None:
            device = torch.cuda.current_device()
        if isinstance(device, (str, int)):
            device = torch.device(device)

        self.need = memory
        self.device = device
        self.hold = hold
        self.exc_time = 0
        self.acc = 5
        self.mem = []
        self.last_success = _memer.get_pid_device_mem(_pid, self.device)

    def copy(self, pid: int, wait: bool = True):
        self.need = _memer.get_pid_device_mem(pid, self.device)
        if wait:
            self.wait(pid)
        else:
            self.start()

    def wait(self, pid):
        while _memer.get_pid_device_mem(pid, self.device) > 0:
            time.sleep(0.5)
        self.start()

    def immediately(self, pre_init=False):
        """
        等待，直到内存有空间后，开始申请相应显存，优雅，礼貌，推荐
        Args:
            pre_init: 是否初始化 CUDA（这将在一开始消耗一定显存），默认为 False，即不抢占任何内存，
                直到设备释放足够空间后开始抢占。
        """
        while True:
            _left = _memer.get_device_release_mem(self.device)
            _allocated = _memer.get_pid_device_mem(_pid, self.device)

            if pre_init and _allocated == 0:
                self._malloc(1, init=True)
                continue

            _need = self.need - _allocated
            if _need < 0:
                return self.end()

            if _left > _need:
                if _allocated == 0:
                    self._malloc(1, init=True)
                    continue

                res = self._malloc(_need)
                time.sleep(0.5)
                if res:
                    return self.end()

            print("need {}Mb, {}Mb allocated, "
                  "waiting for {}Mb released, "
                  "but only {}Mb left.".format(self.need,
                                               _allocated, _need,
                                               _left), end='\r')

    def _malloc(self, size, init=False):
        """ unit: mb """
        try:
            tmp = torch.rand(size, 1048576 // 4, device=self.device)
            if not init:
                self.mem.append(tmp)
            return True
        except Exception as e:
            return False

    def end(self):
        print()
        if self.hold:
            print('press keyboardinterrupt to end')
            try:
                while True:
                    # do some Fake thing
                    self.mem[-1].random_()
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print('continue')

    def start(self, immediately=True):
        if immediately:
            self.immediately()
        else:
            self.invade()

    def invade(self, unit=5):
        """一点一点的侵占，有多少占用多少，直到申请满为止，比较粗鲁，不友好，不推荐"""
        try:
            while self.last_success < self.need:
                res = self._malloc(unit + self.acc)
                if res:
                    self.acc += unit
                    self.last_success = _memer.get_pid_device_mem(_pid, self.device)
                    time.sleep(0.1)
                else:
                    self.exc_time += 1
                    self.acc = max(self.acc - unit, 0)
                    time.sleep(0.5)
                print('{}/{}Mb, try={}, pid={}'.format(self.last_success,
                                                       self.need,
                                                       self.exc_time,
                                                       os.getpid()), end='\r')
            self.end()
        except KeyboardInterrupt:
            print('\nabort.')

    def __enter__(self):
        self.invade()

    def __exit__(self, *args):
        del self.mem[:]
        torch.cuda.empty_cache()
        return True

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad

    @staticmethod
    def hold_current():
        count = torch.cuda.device_count()
        mems = [_memer.get_pid_device_mem(_pid, i) for i in range(count)]
        for i, mem in enumerate(mems):
            if mem > 0:
                memory(mem, device=i, hold=(i == count - 1)).start()
