import torch

from lumo.core import Params
from lumo.core.metaclasses import make_dicts, make_dict
from lumo.exp import SimpleExperiment
from .factory import OptimFactory, InterpFactory


class TrainerExperiment(SimpleExperiment):

    @property
    def log_dir(self):
        return self.test_root

    @property
    def params_fn(self):
        res = self.test_file('params.json')
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
    def saver_dir(self):
        res = self.blob_dir('saver')
        return res

    def dump_train_info(self, epoch: int):
        self.dump_info('trainer', {
            'epoch': epoch
        }, append=True)


class ReimplementExperiment(TrainerExperiment):
    pass


class TrainerPropVar(type):
    def __new__(cls, name, bases, attrs: dict, **kwds):
        for base in bases:
            for key, value in base.__dict__.items():  # type:(str,Any)
                if key.endswith("__"):
                    continue
                if isinstance(value, set):
                    v = attrs.setdefault(key, set())
                    v.update(value)
                elif isinstance(value, dict):
                    v = attrs.setdefault(key, dict())
                    v.update(value)

        clazz = type.__new__(cls, name, bases, dict(attrs))

        make_dicts(clazz, [
            '_prop',
            '_cmp',
            '_rev_index',
            '_call_order',
        ])

        make_dict(clazz, '_state_dicts', {
            'optims': set(),
            'models': set(),
            'others': set(),
            'tensor.th': set(),
            'tensor.np': set(),
        })
        return clazz


class TrainerParams(Params):
    OPTIM = OptimFactory
    SCHE = INTERP = InterpFactory

    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
