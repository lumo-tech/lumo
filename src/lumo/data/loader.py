import warnings
from collections import OrderedDict
from pprint import pformat
from typing import NewType, Union

from torch.utils.data import DataLoader

from lumo.core.metaclasses import PropVar

__all__ = ['DataLoader', 'LumoDataLoader', 'DataLoaderSide', 'DataLoaderType']


class DataLoaderIterWrap:
    def __init__(self, iter_fn, batch_count=None):
        self.iter_fn = iter_fn
        self.iter = iter_fn()
        self.c = 0
        self.batch_count = batch_count

    def __iter__(self):
        while True:
            try:
                yield next(self)
            except StopIteration:
                break

    def __len__(self):
        if self.batch_count is None:
            return len(self.iter)
        else:
            return self.batch_count

    def __next__(self):
        if self.batch_count is not None:
            if self.c >= self.batch_count:
                raise StopIteration()
        try:
            batch = next(self.iter)
        except StopIteration as e:
            if self.batch_count is not None:
                self.iter = self.iter_fn()
                batch = next(self.iter)
            else:
                raise e

        self.c += 1
        return batch


class LumoDataLoader(DataLoader):

    def _wraooed_iter_(self):
        return DataLoaderIterWrap(super().__iter__, len(self))

    def __iter__(self) -> DataLoaderIterWrap:
        return self._wraooed_iter_()


def summarize_loader(loader: DataLoader):
    if isinstance(loader, DataLoaderSide):
        inner = pformat({f"{k}(cycle={loader._cycle[k]})": summarize_loader(v) for k, v in loader._loaders.items()})
        return f"DataLoaderSide({inner})"
    elif isinstance(loader, DataLoader):
        size = '?'
        try:
            size = len(loader)
        except:
            pass
        if loader.batch_sampler is not None:
            batch_size = loader.batch_sampler.batch_size

        # clss = type(loader).__mro__
        # cls_str = []
        # for cls in clss:
        #     if isinstance(cls, DataLoader):
        #         if len(cls_str) == 0:
        #             cls_str.append(cls.__name__)
        #         break
        #     else:
        #         cls_str.append(cls.__name__)
        #
        # cls_str = '|'.join(cls_str)
        return f"{loader.__class__.__name__}(batch_size={batch_size}, num_workers={loader.num_workers}, size={size})"
    else:
        raise ValueError(f'Need {DataLoaderType}, got type {type(loader)}')


class DataLoaderSide:
    """
    `DataLoaderSide` is used when different DataLoader with different batch_size are feeded at the same time.
    """

    def __init__(self):
        self._loaders = OrderedDict()
        self._cycle = OrderedDict()
        self._state = 'zip'

    @property
    def dataset(self):
        return {k: v.dataset for k, v in self.source.items()}

    @property
    def source(self):
        return self._loaders

    def add(self, name, loader: DataLoader, cycle=False):
        self._loaders[name] = loader
        self._cycle[name] = cycle
        return self

    def copy(self):
        loader = DataLoaderSide()
        loader._loaders = self._loaders
        loader._cycle = self._cycle
        loader._state = self._state
        return loader

    def zip(self):
        self._state = 'zip'
        return self

    def chain(self):
        self._state = 'chain'
        return self

    def __len__(self):
        valid_keys = [k for k, cycle in self._cycle.items() if not cycle]
        return min([len(self._loaders[k]) for k in valid_keys])

    def __iter__(self):
        iters = {k: iter(v)
                 for k, v in self._loaders.items()}
        stop = None
        while stop is None:
            res = OrderedDict()
            for k, v in iters.items():
                try:
                    batch = next(v)
                except StopIteration as e:
                    if self._cycle[k]:
                        v = iter(self._loaders[k])
                        iters[k] = v
                        batch = next(v)
                    else:
                        # stop = e
                        return
                res[k] = batch
            if self._state == 'zip':
                yield res
            else:
                yield list(res.values())


DataLoaderType = NewType('DataLoaderType', Union[DataLoader, DataLoaderSide])
