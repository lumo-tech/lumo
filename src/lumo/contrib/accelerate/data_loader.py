from accelerate import synchronize_rng_states, DistributedType
from accelerate.data_loader import DataLoaderDispatcher as _DataLoaderDispatcher, DataLoaderShard as _DataLoaderShard
from accelerate.state import AcceleratorState
from accelerate.utils import send_to_device

from lumo import LumoDataLoader


class DataLoaderDispatcher(_DataLoaderDispatcher):
    pass


class DataLoaderShard(_DataLoaderShard, LumoDataLoader):

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.generator)
        state = AcceleratorState()
        for batch in LumoDataLoader.__iter__(self):
            if state.distributed_type == DistributedType.TPU:
                xm.mark_step()
            yield batch if self.device is None else send_to_device(batch, self.device)
