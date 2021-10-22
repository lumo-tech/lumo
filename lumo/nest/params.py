from ..kit.params import BaseParams
from ..contrib.nn.optimizer_builder import OptimBuilder


class DataLoaderPM(BaseParams):
    def __init__(self):
        super().__init__()
        self.num_workers = 4
        self.batch_size = 12
        self.pin_memory = True


class EvalFirstPM(BaseParams):
    def __init__(self):
        super().__init__()
        self.eval_first = False


class OptimPM(BaseParams):
    def __init__(self):
        super().__init__()
        self.optim = None  # type:OptimBuilder
