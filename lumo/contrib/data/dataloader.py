"""

"""

from torch.utils.data.dataloader import DataLoader as _DataLoader


class DataLoader(_DataLoader):
    """
    DataLoader which allow you reset batch size.

    See
    "Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le, Don't Decay the Learning Rate, Increase the Batch Size"
    https://arxiv.org/abs/1711.00489
        for details
    """

    def set_batch_size(self, batch_size):
        self.batch_sampler.batch_size = batch_size
