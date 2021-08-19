"""

"""
from collections import OrderedDict
from itertools import cycle, chain
from typing import Optional, overload, Union
import torch
from lumo.contrib.itertools import safe_cycle, poll


def identity(x):
    return x


class DataBundler:
    """
    """

    class DataBundler(OrderedDict):
        pass

    def __init__(self, device=None):
        self.dataloaders = DataBundler.DataBundler()
        self.iter_mode = "chain"
        self.device_arg = None
        if device is not None:
            self.to(device)

    def __len__(self):
        if self.iter_mode == 'zip':
            return min(
                len(loader)
                for name, (loader, func) in self.dataloaders.items()
                if func.__name__ != 'safe_cycle'
            )
        elif self.iter_mode == "chain":
            return sum(self.len_list())
        elif self.iter_mode == 'poll':
            return sum(self.len_list())

    def __getitem__(self, item):
        return self.dataloaders[item][0]

    def __iter__(self):
        loaders = self._func_loader()
        if len(loaders) == 1:
            iter = loaders[0]
        elif self.iter_mode == "zip":
            iter = zip(*loaders)
        elif self.iter_mode == "chain":
            iter = chain(*loaders)
        elif self.iter_mode == "poll":
            iter = poll(*loaders)
        else:
            assert False

        for batch_data in iter:
            yield batch_data

    def __repr__(self):
        from pprint import pformat
        return pformat(self.len_dict())

    def _append(self, loader, func, name):
        from torch.utils.data import DataLoader
        assert isinstance(loader, (DataLoader, DataBundler))
        if name is None:
            unname = "unnamed"
            i = 0
            name = "{}_{}".format(unname, i)
            while name in self.dataloaders:
                i += 1
                name = "{}_{}".format(unname, i)
        else:
            assert name not in self.dataloaders, "{} also defined in bundler".format(name)

        self.dataloaders[name] = (loader, func)
        if isinstance(loader, DataBundler) and self.device_arg is not None:
            loader.to(*self.device_arg[0], **self.device_arg[1])

    def _func_loader(self):
        def filter(loader):
            if isinstance(loader, DataBundler):
                return iter(loader)
            return loader

        return [func(filter(loader)) for name, (loader, func) in self.dataloaders.items()]

    def set_batch_size(self, batch_size):
        from torch.utils.data import DataLoader
        from thexp.contrib.data.dataloader import DataLoader as thDataLoader
        for _, (loader, _) in self.dataloaders.items():
            if isinstance(loader, thDataLoader):
                loader.set_batch_size(batch_size)
            elif isinstance(loader, DataLoader):
                loader.batch_sampler.batch_size = batch_size

    def len_list(self):
        """
        按照添加的顺序返回各个dataloader的长度（batch级别）
        :return:
        """
        return [len(loader) for _, (loader, _) in self.dataloaders.items()]

    def len_dict(self):
        """
        返回每个loader的 name:len 字典
        :return: an OrderedDict
        """
        res = DataBundler.DataBundler()
        for name, (loader, func) in self.dataloaders.items():
            res[name] = len(loader)
        return res

    def cycle(self, loader, name=None):
        """

        :param loader: Dataloader object
        :param name:
        :return:
        """
        """一般在zip中保证数据量少的数据集不会成为拖累"""
        self._append(loader, safe_cycle, name)
        return self

    def add(self, loader, name=None):
        self._append(loader, identity, name)
        return self

    @overload
    def to(self, dtype: torch.dtype, non_blocking: bool = False, copy: bool = False):
        ...

    @overload
    def to(self, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None,
           non_blocking: bool = False, copy: bool = False):
        ...

    @overload
    def to(self, other: torch.Tensor, non_blocking: bool = False, copy: bool = False):
        ...

    def to(self, *args, **kwargs):
        if len(args) == 0:
            assert 'device' in kwargs
            device = kwargs['device']
        else:
            device = args[0]
        assert isinstance(device, (torch.device, str))
        self.device_arg = [args, kwargs]
        for (loader, _) in self.dataloaders.values():
            if isinstance(loader, DataBundler):
                loader.to(*args, **kwargs)
        return self

    def choice_batch(self) -> tuple:
        return next(iter(self))

    def zip_mode(self):
        """切换为zip方式提供所有添加的数据集"""
        self.iter_mode = "zip"
        return self

    def chain_mode(self):
        """
        切换为chain方式提供所有添加的数据集
            注意，如果以cycle方法添加了某个数据集，那么该迭代将永远不会停止
        :return:
        """
        self.iter_mode = "chain"
        return self

    def poll_mode(self):
        self.iter_mode = 'poll'
        return self

    @staticmethod
    def create_zip_bundler(**kwargs):
        bundler = DataBundler()
        loaders = [(len(v), v, k) for k, v in kwargs.items()]
        loaders.sort(reverse=True)
        bundler.add(loaders[0][1], loaders[0][2])
        for (_, loader, name) in loaders[1:]:
            bundler.cycle(loader, name)
        return bundler

    @staticmethod
    def create_chain_bundler(*args):
        bundler = DataBundler()
        for loader in args:
            bundler.add(loader)
        return bundler
