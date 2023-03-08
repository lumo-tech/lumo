"""
A module that provides classes and functions to act as the collect_fn in dataloader.

Classes:
    CollateBase: A base class for implementing collate functions.
    IgnoreNoneCollate: A collate function that ignores None samples.

"""
from typing import Mapping, Sequence
from lumo.core.params import ParamsType
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


class CollateBase:
    """
    A base class for implementing collate functions.

    Args:
        params (ParamsType, optional): Parameters for the collate function. Defaults to None.

    Attributes:
        _collate_fn (callable): A function that combines a list of samples into a batch.
        params (ParamsType): Parameters for the collate function.

    Methods:
        from_collate: Creates an instance of the class from a collate function.
        __call__: Applies the collate function to a list of samples.
        before_collate: A hook function called before the collate function is applied.
        raw_collate: Applies the collate function to a list of samples without calling the `before_collate` and `after_collate` hook functions.
        collate: Alias for `raw_collate`.
        after_collate: A hook function called after the collate function is applied.
    """

    @classmethod
    def from_collate(cls, collate_fn, params: ParamsType = None):
        """
        Creates an instance of the class from a collate function.

        Args:
            collate_fn (callable): A function that combines a list of samples into a batch.
            params (ParamsType, optional): Parameters for the collate function. Defaults to None.

        Returns:
            CollateBase: An instance of the class.
        """
        instance = cls(params)
        instance._collate_fn = collate_fn
        return instance

    def __init__(self, params: ParamsType = None) -> None:
        super().__init__()
        self._collate_fn = default_collate
        self.params = params

    def __call__(self, *args, **kwargs):
        """
        Applies the collate function to a list of samples.

        Args:
            *args: A list of samples.
            **kwargs: Additional keyword arguments.

        Returns:
            The batch of samples.
        """
        res = self.before_collate(*args, **kwargs)
        res = self._collate_fn(res)
        res = self.after_collate(res)
        return res

    def before_collate(self, sample_list):
        """
        A hook function called before the collate function is applied.

        Args:
            sample_list (Sequence): A list of samples.

        Returns:
            Sequence: The modified list of samples.
        """
        return sample_list

    def raw_collate(self, sample_list):
        """
        Applies the collate function to a list of samples without calling the `before_collate` and `after_collate` hook functions.

        Args:
            sample_list (Sequence): A list of samples.

        Returns:
            The batch of samples.
        """
        return self._collate_fn(sample_list)

    def collate(self, sample_list):
        """
        Alias for `raw_collate`.

        Args:
            sample_list (Sequence): A list of samples.

        Returns:
            The batch of samples.
        """
        return self._collate_fn(sample_list)

    def after_collate(self, batch):
        """
        A hook function called after the collate function is applied.

        Args:
            batch (Any): The batch of samples.

        Returns:
            Any: The modified batch of samples.
        """
        return batch


class IgnoreNoneCollate(CollateBase):
    """A collate function that ignores `None` samples."""

    def _filter_none(self, item):
        if item is None:
            return False
        if isinstance(item, (list, tuple)):
            return all([self._filter_none(i) for i in item])
        if isinstance(item, dict):
            return all(self._filter_none(i) for i in item.values())
        return True

    def before_collate(self, sample_list):
        """ before collate"""
        return list(filter(self._filter_none, sample_list))


def numpy_collate(batch):
    """Collate function for numpy arrays.

    Args:
        batch (list): A list of numpy arrays or other python objects.

    Returns:
        numpy.ndarray or dict or list: Returns a collated numpy array, a dictionary of collated numpy arrays,
        or a list of collated numpy arrays depending on the type of input elements.

    Raises:
        RuntimeError: If the elements in batch do not have consistent size.

    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        batch = [elem.detach().cpu().numpy() for elem in batch]
        return np.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return numpy_collate([np.array(b) for b in batch])
        elif elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(elem, float):
        return np.array(batch, dtype=np.float)
    elif isinstance(elem, int):
        return np.array(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, Mapping):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(numpy_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
