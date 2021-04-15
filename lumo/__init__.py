"""

"""
__version__ = "0.1.1.20"

from .utils.paths import global_config_path as _
from .utils.keys import K

from .kit import (
    F, Q,
    BaseParams, Params,
    Meter, AvgMeter,

    DatasetBuilder, DataDelegate, DataModule,
    Saver, Random,

    TrainerExperiment, exphook,
    globs, Trainer, callbacks
)

from . import calculate  # initialize schedule attr classes

_()  # check and initialize global config
