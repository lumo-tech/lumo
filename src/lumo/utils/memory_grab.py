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
    """
    A class that represents device memory usage.
    """
    def __init__(self):
        self.line_mem = tree()

    def _parse_device_pid_mem_pair(self, lines):
        """
        Parses device ID, process ID, and memory usage (in MiB) from the given list of strings.

        Args:
            lines (List[str]): List of strings to parse.

        Yields:
            Tuple[int, int, int]: Tuple containing device ID, process ID, and memory usage (in MiB) for each match found.
        """
        for lid, line in enumerate(lines):
            res = re.search(match_mem, line)
            if res is not None:
                _device, _pid, _mib = [int(i) for i in res.groups()]
                self.line_mem[_device][_pid] = lid
                yield _device, _pid, _mib

    def try_parse(self, lines, pid, device):
        """
        Attempts to parse memory usage (in MiB) for a process running on a specific device using the cached line ID.

        Args:
            lines (List[str]): List of strings to parse.
            pid (int): Process ID to look for.
            device (int or str or torch.device): Device ID to look for.

        Returns:
            int: Memory usage in MiB for the specified process and device if found, -1 otherwise.
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
        """
        Parses memory usage (in MiB) for a process running on a specific device by searching through the list of strings.

        Args:
            lines (List[str]): List of strings to parse.
            pid (int): Process ID to look for.
            device (int or str or torch.device): Device ID to look for.

        Returns:
            int: Memory usage in MiB for the specified process and device if found, 0 otherwise.
        """
        _res = self.try_parse(lines, pid, device)
        if _res != -1:
            return _res

        for _device, _pid, _mib in self._parse_device_pid_mem_pair(lines):
            if _pid == pid and _device == device:
                return _mib

        return 0

    def _get_nvidia_smi(self):
        """
        Executes the 'nvidia-smi' command and returns the output as a list of strings.

        Returns:
            List[str]: List of strings representing the output of the 'nvidia-smi' command.
        """
        proc = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
        lines = proc.stdout.readlines()
        return [i.decode() for i in lines]

    def _device_equal(self, da, db):
        """
        Compares two device IDs or names for equality.

        Args:
            da (int or str or torch.device): First device ID or name to compare.
            db (int or str or torch.device): Second device ID or name to compare.

        Returns:
            bool: True if the two device IDs or names are equal, False otherwise.
        """
        if isinstance(da, (int, str)):
            da = torch.device(da)
        if isinstance(db, (int, str)):
            db = torch.device(db)
        return da == db

    def get_device_release_mem(self, device):
        """
        Returns the amount of free memory (in MiB) on a specified device.

        Args:
            device (int or str or torch.device): Device ID or name to look up.

        Returns:
            int: Amount of free memory (in MiB) on the specified device.
        """
        s_pid = os.getpid()
        total = self.get_device_mem(device)
        for _device, _pid, _mib in self._parse_device_pid_mem_pair(self._get_nvidia_smi()):
            if self._device_equal(device, _device):
                total -= _mib

        return total

    def get_device_mem(self, device):
        """
        Returns the total amount of memory (in MiB) on a specified device.

        Args:
            device (int or str or torch.device): Device ID or name to look up.

        Returns:
            int: Total amount of memory (in MiB) on the specified device.
        """
        return torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)

    def get_pid_device_mem(self, pid, device):
        """
        Attempts to obtain the memory usage (in MiB) for a specific process running on a specific device.

        Args:
            pid (int): Process ID to look up.
            device (int or str or torch.device): Device ID or name to look up.

        Returns:
            int: Memory usage in MiB for the specified process and device if found, -1 otherwise.
        """
        if isinstance(device, torch.device):
            device = device.index
        else:
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
    """
    A graceful memory allocator that optimizes GPU memory allocation by incrementally increasing the memory
    footprint to minimize fragmentation.

    ```python
    import lumo
    with lumo.memory(5000):
        y = x * 2

    @lumo.memory(1024)
    def doubler(x):
        ...

    lumo.memory(10000).start()
    ...
    ```

    Args:
        memory (int): Memory size to be allocated in MB.
        device (str or int, optional): Device to allocate memory on. Defaults to the current CUDA device.
        hold (bool, optional): Whether to hold the memory after allocation. Defaults to False.
        invade (bool, optional): Whether to use aggressive memory allocation. Defaults to False.



    References:
        To get GPU memory usage, we use nvidia-smi. Refer to this link for details:
        https://github.com/pytorch/pytorch/issues/12873
    """

    def __init__(self, memory, device=None, hold=False, invade=False) -> None:
        """
        Initialize the memory allocator.

        Args:
            memory (int): Memory size to be allocated in MB.
            device (str or int, optional): Device to allocate memory on. Defaults to the current CUDA device.
            hold (bool, optional): Whether to hold the memory after allocation. Defaults to False.
            invade (bool, optional): Whether to use aggressive memory allocation. Defaults to False.
        """
        super().__init__()
        if device is None:
            device = torch.cuda.current_device()
        if isinstance(device, (str, int)):
            device = torch.device(device)

        self.need = memory
        self.device = device
        self.hold = hold
        self.is_invade = invade
        self.exc_time = 0
        self.acc = 5
        self.mem = []
        self.last_success = _memer.get_pid_device_mem(_pid, self.device)

    def copy(self, pid: int, wait: bool = True):
        """
        Copy memory allocation parameters from another process.

        Args:
            pid (int): Process ID to copy memory parameters from.
            wait (bool, optional): Whether to wait until the other process has finished before starting allocation.
            Defaults to True.
        """
        self.need = _memer.get_pid_device_mem(pid, self.device)
        if wait:
            self.wait(pid)
        else:
            self.start()

    def wait(self, pid):
        """
        Wait for the other process to finish before starting allocation.

        Args:
            pid (int): Process ID to wait for.
        """
        while _memer.get_pid_device_mem(pid, self.device) > 0:
            time.sleep(0.5)
        self.start()

    def immediately(self, pre_init=False):
        """
        Wait until there is enough memory available, then allocate the necessary memory.
        This is the recommended way to allocate memory.

        Args:
            pre_init (bool, optional): Whether to initialize CUDA (which will consume a certain amount of memory)
            before allocating. Defaults to False, meaning memory will only be allocated after enough memory is
            released by other processes.
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
        """
        Allocate memory of the given size (in MB) on the specified device.

        Args:
            size (int): Memory size to be allocated, in MB.
            init (bool, optional): Whether to initialize CUDA. Defaults to False.

        Returns:
            bool: True if the memory allocation was successful, False otherwise.
        """
        try:
            tmp = torch.rand(size, 1048576 // 4, device=self.device)
            if not init:
                self.mem.append(tmp)
            return True
        except Exception as e:
            return False

    def end(self):
        """
        Release allocated memory and empty the CUDA cache.
        """
        del self.mem[:]
        torch.cuda.empty_cache()

    def start(self, immediately=True):
        """
        Start memory allocation.

        Args:
            immediately (bool, optional): Whether to use the recommended memory allocation method.
            Defaults to True.
        """
        if immediately:
            self.immediately()
        else:
            self.invade()

    def invade(self, unit=5):
        """
        Aggressively allocate memory, increasing the memory footprint until the necessary amount is allocated.
        This method is not recommended.

        Args:
            unit (int, optional): Incremental size of memory allocation (in MB). Defaults to 5.
        """
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
        """
         Start memory allocation when entering the 'with' block.
        """
        self.start(immediately=not self.is_invade)

    def __exit__(self, *args):
        """
        Release memory and empty the CUDA cache when exiting the 'with' block.

        Returns:
            bool: Always returns True.
        """
        self.end()
        return True

    def __call__(self, func):
        """
        Decorator to use with functions that require memory allocation.

        Args:
            func (callable): The function to decorate.

        Returns:
            callable: The decorated function.
        """

        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            """decorate"""
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad

    @staticmethod
    def hold_current():
        """
        Hold the currently allocated memory.
        """
        count = torch.cuda.device_count()
        mems = [_memer.get_pid_device_mem(_pid, i) for i in range(count)]
        for i, mem in enumerate(mems):
            if mem > 0:
                memory(mem, device=i, hold=(i == count - 1)).start()
