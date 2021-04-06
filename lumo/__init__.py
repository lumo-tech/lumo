"""

"""
__version__ = "1.5.4.4"

from .utils.paths import global_config_path as _

from .kit import (
    BaseParams, Params,
    Meter, AvgMeter,

    DataBridge, DatasetBuilder, DataDelegate,
    Saver, Random,

    Trainer, DistributedTrainer, callbacks,
    TrainerExperiment, exphook,
    globs
)

from . import calculate  # initialize schedule attr classes

_()  # check and initialize global config
