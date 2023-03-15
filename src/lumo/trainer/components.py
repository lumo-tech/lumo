import torch

from lumo.core import Params
from lumo.exp import SimpleExperiment
from .factory import OptimFactory, InterpFactory


class TrainerExperiment(SimpleExperiment):
    """A class for helping manage an experiment by Trainer."""

    @property
    def log_dir(self):
        return self.info_dir

    @property
    def params_fn(self):
        res = self.mk_ipath('params.yaml')
        return res

    @property
    def board_args(self):
        key = 'tensorboard_args'
        if self.has_prop(key):
            return self.get_prop(key)
        else:
            log_dir = self.mk_ipath('board', is_dir=True)
            res = {
                'filename_suffix': '.bd',
                'log_dir': log_dir,
            }
            self.dump_info('tensorboard_args', res)
            return res

    @property
    def state_dict_dir(self):
        res = self.mk_bpath('state_dict', is_dir=True)
        return res

    def dump_train_eidx(self, eidx, epoch: int):
        """
        Dumps the progress of the trainer.

         Args:
            eidx (int): The index of the current epoch (starting from 0).
            epoch (int): The total number of epochs to train for.
        """
        self.dump_progress((eidx + 1) / epoch, update_from='trainer')


class TrainerParams(Params):
    """
    A class to hold parameters for trainer.
    """
    OPTIM = OptimFactory
    SCHE = INTERP = InterpFactory

    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
