from accelerate import synchronize_rng_states, DistributedType
from accelerate.data_loader import DataLoaderDispatcher as _DataLoaderDispatcher, DataLoaderShard as _DataLoaderShard
from accelerate.state import AcceleratorState
from accelerate.utils import send_to_device

from lumo import LumoDataLoader


class DataLoaderDispatcher(_DataLoaderDispatcher):
    pass
