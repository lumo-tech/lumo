from collections import OrderedDict
from typing import NewType, Union

from torch.utils.data import DataLoader

from lumo.core.metaclasses import PropVar

__all__ = ['DataLoader', 'LumoDataLoader', 'DataLoaderSide']


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


class LumoDataLoader(DataLoader, metaclass=PropVar):

    def set_prop(self, prop):
        return self._prop.update(prop)

    def set_batch_count(self, size):
        self._prop['batch_count'] = size
        return self

    def __len__(self):
        bc = self._prop.get('batch_count', None)
        if bc is None:
            return super(LumoDataLoader, self).__len__()
        return bc

    def __iter__(self) -> DataLoaderIterWrap:
        return DataLoaderIterWrap(super().__iter__,
                                  self._prop.get('batch_count', None))


def summarize_loader(loader: DataLoader):
    if loader.batch_sampler is not None:
        batch_size = loader.batch_sampler.batch_size
    return f"DataLoader(batch_size={batch_size}, num_workers={loader.num_workers})"


class DataLoaderSide:
    """
    `DataLoaderSide` is used when different DataLoader with different batch_size are feeded at the same time.
    """

    def __init__(self):
        self._loaders = OrderedDict()
        self._cycle = OrderedDict()
        self._state = 'zip'

    def add(self, name, loader: DataLoader, cycle=False):
        self._loaders[name] = loader
        self._cycle[name] = cycle
        return self

    def zip(self):
        self._state = 'zip'
        return self

    def chain(self):
        self._state = 'chain'
        return self

    def __len__(self):
        pass

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
                        stop = e
                        break
                res[k] = batch
            if self._state == 'zip':
                yield res
            else:
                yield list(res.values())


DataLoaderType = NewType('DataLoaderType', Union[DataLoader, DataLoaderSide])
