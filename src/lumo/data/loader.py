from collections import OrderedDict
from pprint import pformat
from typing import NewType, Union

from torch.utils.data import DataLoader


class LumoDataLoader(DataLoader):
    """This module defines the LumoDataLoader class that inherits from the DataLoader class."""
    pass


def summarize_loader(loader: DataLoader):
    """
    Summarize the DataLoader object and return a formatted string representation.

    Args:
        loader: A DataLoader object.

    Returns:
        A formatted string representation of the DataLoader object.

    Raises:
        ValueError: If the input argument is not a DataLoader object.

    """
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


        return f"{loader.__class__.__name__}(batch_size={batch_size}, num_workers={loader.num_workers}, size={size})"
    else:
        raise ValueError(f'Need {DataLoaderType}, got type {type(loader)}')


class DataLoaderSide:
    """
    A utility class for loading data from different DataLoaders with different batch sizes at the same time.

    Example usage:

    ```python
    from lumo import DataLoaderSide
    loader = DataLoaderSide()
    loader.add('train', train_loader, cycle=True)
    loader.add('val', val_loader)
    loader.zip()
    for batch in loader:
        # process batch
    ```

    Methods:
        dataset(): Returns a dictionary that maps the name of the DataLoader to its corresponding dataset.
        source(): Returns the _loaders dictionary.
        add(name, loader, cycle=False): Adds a DataLoader instance to the _loaders dictionary.
        name is the name of the DataLoader.
        loader is the DataLoader instance to be added.
        cycle is a boolean indicating whether the DataLoader should be cycled. Defaults to False.
        copy(): Returns a new DataLoaderSide instance with the same _loaders, _cycle, and _state attributes as the original.
        zip(): Sets the _state attribute to 'zip', which means the batches are zipped together.
            if _state is 'zip', the batches are returned as an ordered dictionary.
        chain(): Sets the _state attribute to 'chain', which means the batches are concatenated.
            if _state is 'chain', the batches are returned as a list.
        len(): Returns the minimum length of all the DataLoaders that do not have the cycle flag set to True.
        iter(): Returns an iterator that generates batches from the DataLoaders in the _loaders dictionary.

    """

    def __init__(self):
        self._loaders = OrderedDict()
        self._cycle = OrderedDict()
        self._state = 'zip'

    @property
    def dataset(self):
        """Returns a dictionary that maps the name of the DataLoader to its corresponding dataset."""
        return {k: v.dataset for k, v in self.source.items()}

    @property
    def source(self):
        """Returns the _loaders dictionary."""
        return self._loaders

    def add(self, name, loader: DataLoader, cycle=False):
        """
        Adds a DataLoader instance to the _loaders dictionary.
        Args:
            name (str): The name of the DataLoader.
            loader (DataLoader): The DataLoader instance to be added.
            cycle (bool): A boolean indicating whether the DataLoader should be cycled. Defaults to False.

        """
        self._loaders[name] = loader
        self._cycle[name] = cycle
        return self

    def copy(self):
        """Returns a new DataLoaderSide instance with the same _loaders, _cycle, and _state attributes as the original."""
        loader = DataLoaderSide()
        loader._loaders = self._loaders
        loader._cycle = self._cycle
        loader._state = self._state
        return loader

    def zip(self):
        """Sets the _state attribute to 'zip', which means the batches are zipped together.
        If _state is 'zip', the batches are returned as an ordered dictionary."""
        self._state = 'zip'
        return self

    def chain(self):
        """
        Sets the _state attribute to 'chain', which means the batches are concatenated.
            If _state is 'chain', the batches are returned as a list.
        """
        self._state = 'chain'
        return self

    def __len__(self):
        """Returns the minimum length of all the DataLoaders that do not have the cycle flag set to True."""
        valid_keys = [k for k, cycle in self._cycle.items() if not cycle]
        return min([len(self._loaders[k]) for k in valid_keys])

    def __iter__(self):
        """Returns an iterator that generates batches from the DataLoaders in the _loaders dictionary."""
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
