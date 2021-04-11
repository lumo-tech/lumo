"""

"""
__version__ = "0.1.0"

from .utils.paths import global_config_path as _

from .kit import (
    BaseParams, Params,
    Meter, AvgMeter,

    DataBridge, DatasetBuilder, DataDelegate,
    Saver, Random,

    TrainerExperiment, exphook,
    globs
)

from . import calculate  # initialize schedule attr classes

_()  # check and initialize global config
