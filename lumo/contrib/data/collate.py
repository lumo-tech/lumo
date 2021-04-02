"""

"""
from torch.utils.data._utils.collate import default_collate as collate


class CollateFn():
    def __init__(self, default_collate=collate):
        self._default_collate = collate

    def before_collate(self, sample_list, mem: dict):
        return sample_list

    def after_collate(self, batch, mem: dict):
        return batch

    def __call__(self, batch):
        mem = {}
        batch = self.before_collate(batch, mem)
        batch = self._default_collate(batch)
        batch = self.after_collate(batch, mem)
        return batch
