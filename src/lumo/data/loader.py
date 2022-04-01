from torch.utils.data import DataLoader

from lumo.core.metaclasses import PropVar

__all__ = ['DataLoader', 'LumoDataLoader']


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
