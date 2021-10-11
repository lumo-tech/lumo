from dataclasses import dataclass

import numpy as np


# from lumo.utils.fmt import to_ndarray, validate_scalar_shape


@dataclass()
class Scalar:
    tag: str
    value: np.ndarray
    global_step: int = None
    walltime: float = None
