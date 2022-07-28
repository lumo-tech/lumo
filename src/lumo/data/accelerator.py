from typing import Optional, List, Union

import torch
from accelerate.data_loader import (IterableDatasetShard, BatchSamplerShard, _PYTORCH_DATALOADER_KWARGS,
                                    DataLoaderDispatcher as _DataLoaderDispatcher, DataLoaderShard as _DataLoaderShard)
from accelerate.state import AcceleratorState, DistributedType, is_tpu_available
from accelerate.utils import RNGType, synchronize_rng_states, send_to_device
from torch.utils.data import DataLoader, IterableDataset
from .loader import LumoDataLoader, DataLoaderIterWrap

if is_tpu_available():
    import torch_xla.core.xla_model as xm


class DataLoaderDispatcher(_DataLoaderDispatcher, LumoDataLoader):

    def __iter__(self) -> DataLoaderIterWrap:
        return super().__iter__()


class DataLoaderShard(_DataLoaderShard, LumoDataLoader):

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.generator)
        state = AcceleratorState()
        for batch in LumoDataLoader.__iter__(self):
            if state.distributed_type == DistributedType.TPU:
                xm.mark_step()
            yield batch if self.device is None else send_to_device(batch, self.device)


def prepare_data_loader(
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        num_processes: Optional[int] = None,
        process_index: Optional[int] = None,
        split_batches: bool = False,
        put_on_device: bool = False,
        rng_types: Optional[List[Union[str, RNGType]]] = None,
        dispatch_batches: Optional[bool] = None,
) -> DataLoader:
    """
    Wraps a PyTorch :obj:`DataLoader` to generate batches for one of the processes only.

    Depending on the value of the :obj:`drop_last` attribute of the :obj:`dataloader` passed, it will either stop the
    iteration at the first batch that would be too small / not present on all processes or loop with indices from the
    beginning.

    Args:
        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            The data loader to split across several devices.
        device (:obj:`torch.device`):
            The target device for the returned :obj:`DataLoader`.
        num_processes (:obj:`int`, `optional`):
            The number of processes running concurrently. Will default to the value given by
            :class:`~accelerate.AcceleratorState`.
        process_index (:obj:`int`, `optional`):
            The index of the current process. Will default to the value given by :class:`~accelerate.AcceleratorState`.
        split_batches (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the resulting :obj:`DataLoader` should split the batches of the original data loader across devices
            or yield full batches (in which case it will yield batches starting at the :obj:`process_index`-th and
            advancing of :obj:`num_processes` batches at each iteration).

            Another way to see this is that the observed batch size will be the same as the initial :obj:`dataloader`
            if this option is set to :obj:`True`, the batch size of the initial :obj:`dataloader` multiplied by
            :obj:`num_processes` otherwise.

            Setting this option to :obj:`True` requires that the batch size of the :obj:`dataloader` is a round
            multiple of :obj:`batch_size`.
        put_on_device (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to put the batches on :obj:`device` (only works if the batches are nested list, tuples or
            dictionaries of tensors).
        rng_types (list of :obj:`str` or :class:`~accelerate.utils.RNGType`):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - :obj:`"torch"`: the base torch random number generator
            - :obj:`"cuda"`: the CUDA random number generator (GPU only)
            - :obj:`"xla"`: the XLA random number generator (TPU only)
            - :obj:`"generator"`: the :obj:`torch.Generator` of the sampler (or batch sampler if there is no sampler in
              your dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

        dispatch_batches (:obj:`bool`, `optional`):
            If set to :obj:`True`, the datalaoder prepared is only iterated through on the main process and then the
            batches are split and broadcast to each process. Will default to :obj:`True` when the underlying dataset is
            an :obj:`IterableDataset`, :obj:`False` otherwise.

    Returns:
        :obj:`torch.utils.data.dataloader.DataLoader`: A new data loader that will yield the portion of the batches

    .. warning::

        This does not support :obj:`BatchSampler` with varying batch size yet.
    """
    if dispatch_batches is None:
        dispatch_batches = False if not put_on_device else isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")
    # Grab defaults from AcceleratorState
    state = AcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    # Sanity check
    if split_batches and dataloader.batch_size > 1 and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
            f"needs to be a round multiple of the number of processes ({num_processes})."
        )

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    generator = getattr(dataloader, "generator", None)
    # No change if no multiprocess
    if num_processes != 1 and not dispatch_batches:
        if isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                generator = dataloader.dataset.generator
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            # New batch sampler for the current process.
            if hasattr(dataloader.sampler, "generator"):
                if dataloader.sampler.generator is None:
                    dataloader.sampler.generator = torch.Generator()
                    generator = dataloader.sampler.generator
                    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            elif getattr(dataloader.batch_sampler, "generator", None) is not None:
                generator = dataloader.batch_sampler.generator
            new_batch_sampler = BatchSamplerShard(
                dataloader.batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
        "generator",
    ]

    if rng_types is not None and generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["batch_size"] = dataloader.batch_size // num_processes if split_batches else dataloader.batch_size

    if dispatch_batches:
        res = DataLoaderDispatcher(
            new_dataset, split_batches=split_batches, batch_sampler=new_batch_sampler, **kwargs
        )
    else:
        res = DataLoaderShard(
            new_dataset,
            device=device if put_on_device else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            generator=generator,
            **kwargs,
        )

    if isinstance(dataloader, LumoDataLoader):
        res.set_prop(dataloader._prop)

    return res
