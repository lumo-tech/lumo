from typing import NewType

import torch

from lumo.core import Params
from lumo.exp import SimpleExperiment
from .factory import OptimFactory, InterpFactory


class TrainerExperiment(SimpleExperiment):

    @property
    def log_dir(self):
        return self.test_root

    @property
    def params_fn(self):
        res = self.test_file('params.yaml')
        self.dump_string('params.yaml', res)
        return res

    @property
    def board_args(self):
        key = 'tensorboard_args'
        if self.has_prop(key):
            return self.get_prop(key)
        else:
            log_dir = self.test_dir('board')
            res = {
                'filename_suffix': '.bd',
                'log_dir': log_dir,
            }
            self.dump_info('tensorboard_args', res)
            return res

    @property
    def state_dict_dir(self):
        res = self.blob_dir('state_dict')
        return res

    def dump_train_eidx(self, eidx, epoch: int):
        """
        Args:
            eidx: start from 0, end at `epoch-1`
            epoch:
        """
        self.dump_progress((eidx + 1) / epoch, update_from='trainer')


class ReimplementExperiment(TrainerExperiment):
    pass


class TrainerParams(Params):
    OPTIM = OptimFactory
    SCHE = INTERP = InterpFactory

    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
