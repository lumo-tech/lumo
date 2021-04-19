from typing import Optional, Sequence

from torch.utils.data import dataloader as _loader, Dataset, Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t


class DataLoader(_loader.DataLoader):
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: bool = False,
                 sampler: Optional[Sampler[int]] = None, batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0, collate_fn: _collate_fn_t = None, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0, worker_init_fn: _worker_init_fn_t = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    def _get_iterator(self) -> _loader._BaseDataLoaderIter:
        return super()._get_iterator()

    @property
    def multiprocessing_context(self):
        return super().multiprocessing_context()

    def __setattr__(self, attr, val):
        return super().__setattr__(attr, val)

    def __iter__(self) -> _loader._BaseDataLoaderIter:
        return super().__iter__()

    @property
    def _auto_collation(self):
        return super()._auto_collation()

    @property
    def _index_sampler(self):
        return super()._index_sampler()

    def __len__(self) -> int:
        return super().__len__()
