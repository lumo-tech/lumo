"""

"""

from torch.utils.data.dataloader import DataLoader as _DataLoader

# from thexp.utils.decorators.deprecated import deprecated
from thexp import __VERSION__


class DataLoader(_DataLoader):
    """
    用于可调整 batch_size 的DataLoader，其需求基于

    Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le, Don't Decay the Learning Rate, Increase the Batch Size
    https://arxiv.org/abs/1711.00489
    """

    def set_batch_size(self, batch_size):
        self.batch_sampler.batch_size = batch_size

    def DataBundler(self, add=True, name=None):
        from thexp import DataBundler
        bundler = DataBundler()
        if add:
            return bundler.add(self, name)
        else:
            return bundler.cycle(self, name)
