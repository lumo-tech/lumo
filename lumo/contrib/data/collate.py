"""

"""
from functools import wraps
from typing import Any, Mapping

from torch.utils.data._utils.collate import default_collate


class CollateBase():

    def __new__(cls, *args, **kwargs) -> Any:
        self = super().__new__(cls)

        def wrap(func):
            @wraps(func)
            def inner(*args, **kwargs):
                res = self.before_collate(*args, **kwargs)
                res = func(res)
                res = self.after_collate(res)
                return res

            return inner

        self.collate = wrap(self.collate)
        return self

    def __call__(self, *args, **kwargs):
        return self.collate(*args, **kwargs)

    def before_collate(self, sample_list):
        return sample_list

    def collate(self, sample_list):
        return default_collate(sample_list)

    def after_collate(self, batch):
        return batch


class IgnoreNoneCollate(CollateBase):

    def _filter_none(self, item):
        if item is None:
            return False
        if isinstance(item, (list, tuple)):
            return all([self._filter_none(i) for i in item])
        if isinstance(item, dict):
            return all(self._filter_none(i) for i in item.values())
        return True

    def before_collate(self, sample_list):
        return list(filter(self._filter_none, sample_list))
