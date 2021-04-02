"""

"""
import bisect
import inspect
import os
from collections.abc import Iterator
from functools import wraps
from typing import Any, Dict, TypeVar

# from ..utils.lazy import torch, np
import numpy as np
import torch
from torch import distributed as dist
from torch.optim.optimizer import Optimizer

from lumo.base_classes import attr
from lumo.base_classes.metaclasses import Merge
from lumo.kit.logger import Logger
from lumo.kit.builder import Dataset
from lumo.kit.environ import globs
from lumo.kit.experiment import TrainerExperiment
from lumo.kit.meter import Meter
from lumo.kit.params import Params, BaseParams
from lumo.utils.keys import TRAINER

ParamsType = TypeVar('ParamsType', bound=Params)


def _gene_class_exp_name(trainer_instance) -> str:
    try:
        file = inspect.getfile(trainer_instance.__class__)
        pre = os.path.splitext(os.path.basename(file))[0]
    except:
        pre = 'builtin'

    return "{}.{}".format(pre, trainer_instance.__exp_name__)


def _gene_trainer_info(trainer_instance) -> dict:
    try:
        path = inspect.getfile(trainer_instance.__class__)
        basefn = os.path.splitext(os.path.basename(path))[0]
    except:
        path = 'built_in'
        basefn = 'built_in'
    from lumo.utils.keys import TRAINER
    trainer_kwargs = {
        TRAINER.path: path,
        TRAINER.doc: trainer_instance.__class__.__doc__,
        TRAINER.basename: basefn,
        TRAINER.class_name: trainer_instance.__class__.__name__
    }

    return trainer_kwargs


def mp_agent(rank, trainer, op):
    import torch.distributed as dist
    trainer.params.local_rank = rank
    dist.init_process_group(backend='nccl', init_method=trainer.params.init_method,
                            rank=rank,
                            world_size=trainer.params.world_size)

    trainer.params.device = 'cuda:{}'.format(rank)
    trainer.regist_device(torch.device(trainer.params.device))
    torch.cuda.set_device(trainer.params.local_rank)
    print('in rank {}'.format(rank))
    trainer.models(trainer.params)
    trainer.datasets(trainer.params)
    trainer.callbacks(trainer.params)
    op(trainer)


class TArg(BaseParams):
    def __init__(self):
        super().__init__()


class BaseTrainer(metaclass=Merge):
    """
    1. 组装所有的插件
    2. 提供完整的训练流程 api
    3. 提供对需要 api 函数的 callback


    Trainer helps you to control all stage happened in deeplearning, including train/evaluation/test, save/load checkpoints
    , log/meter/tensorboard variable, etc. It's also very convenient to custom your own logits.

    Thanks for git, trainer will commit all change on `experiments` branch before you start your training, and
    this makes you easily



    When create a Trainer instance, a Params instance should be passed
    """

    __exp_name__ = None
    _call_backs = {
        "initial",
        "train", "train_epoch", "train_step", "test", "eval", "train_on_batch",
        "regist_databundler", "train_batch", "test_eval_logic", "test_eval_logic_v2", "predict",
        "load_keypoint", "load_checkpoint", "load_model", "save_keypoint", "save_checkpoint", "save_model",
    }

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if cls.__exp_name__ is None:
            cls.__exp_name__ = cls.__name__.lower().replace("trainer", "Exp")

        def wrapper(func, _call_set: list):
            """
            对每个 Trainer 的 _call_backs 类变量中定义的函数尝试绑定回调
            Args:
                func:
                _call_set:

            Returns:

            """

            @wraps(func)
            def _newfunc(*aargs, **kkwargs):
                """执行前回调 on_begin() 、执行后回调 on_end()、执行异常则回调 on_exception() """
                for callback in _call_set:
                    callback.on_begin(self, func, self.params, *aargs, **kkwargs)
                try:
                    _meter = func(*aargs, **kkwargs)
                    # 如果返回迭代器，那么会消耗掉迭代器，并只返回最后一次运行的结果（如果有）
                    if isinstance(_meter, Iterator):
                        _m = Meter()
                        for _m in _meter:
                            pass
                        _meter = _m
                except BaseException as e:
                    _handles = [callback.on_exception(self, func, self.params, e, *aargs, **kkwargs)
                                for callback in _call_set]

                    if any(_handles):
                        return None
                    else:
                        # TODO 添加 rich 输出，更优雅的退出
                        raise e

                for callback in _call_set:
                    callback.on_end(self, func, self.params, _meter, *aargs, **kkwargs)
                return _meter

            return _newfunc

        self._callback_set = []
        self._callback_name_set = set()

        vars = dir(self)
        for name in vars:
            if name not in self._call_backs:
                continue
            if name.startswith("_"):
                continue
            value = getattr(self, name, None)
            if value is None:
                continue
            if callable(value):
                setattr(self, name, wrapper(value, self._callback_set))
        return self

    def __init__(self, params: ParamsType):
        self._state_dicts = attr({
            "models": {},
            "optims": {},
            "buffers": {},
            "others": {},
            'device': torch.device('cpu'),
            'devices': {},
            'initial': {
                'models': False,
                'datasets': False,
                'callbacks': False,
            },
            'train_epoch_toggle': False,
            'train_toggle': False,
            'params' : params,
            'exp' : TrainerExperiment(_gene_class_exp_name(self)),
        })
        self._datasets = attr()  # type:Dict[str,Dataset]

        # self.experiment = TrainerExperiment(_gene_class_exp_name(self))  # type: TrainerExperiment

        if 'device' in params:
            _device = torch.device(params.device)
            self.regist_device(_device)

        self._initialize_globs()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        from torch.optim.optimizer import Optimizer
        if isinstance(value, torch.device):
            self._state_dicts[TRAINER.DKEY.devices][name] = value
        elif isinstance(value, torch.nn.Module):
            self._state_dicts[TRAINER.DKEY.models][name] = value
        elif isinstance(value, Optimizer):
            self._state_dicts[TRAINER.DKEY.optims][name] = value
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            self._state_dicts[TRAINER.DKEY.buffers][name] = value
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            self._state_dicts[TRAINER.DKEY.others][name] = value

    def __setitem__(self, key: str, value: Any):
        self.__setattr__(key, value)

    def __setstate__(self, state):
        self._state_dicts = attr.from_dict(state)

    def __getstate__(self):
        res = self._state_dicts.pickify()
        return res

    def _initialize_globs(self):
        if dist.is_initialized():
            globs['rank'] = dist.get_rank()
            globs['world_size'] = dist.get_world_size()
        else:
            globs['rank'] = -1
            globs['world_size'] = 0

        for k, v in self.params.items():  # type:str,Any
            if k.isupper():
                globs[k] = v
                if isinstance(v, str):
                    os.environ[k] = v
    @property
    def exp(self)->TrainerExperiment:
        return self._state_dicts[TRAINER.DKEY.exp]

    @property
    def logger(self):
        if self._logger is None:
            self._logger = Logger()
            self._logger.add_log_dir(self.exp.log_dir)
        return self._logger



    @property
    def params(self)->ParamsType:
        return self._state_dicts[TRAINER.DKEY.params]

    @property
    def device(self) -> torch.device:
        return self._state_dicts[TRAINER.DKEY.device]

    def regist_device(self, device: torch.device):
        self._state_dicts[TRAINER.DKEY.device] = device
        if device.type == 'cuda':
            torch.cuda.set_device(device)

    def regist_dataset(self, train: Dataset = None, val: Dataset = None, test: Dataset = None,
                       **others: Dict[str, Dataset]):
        pass

    def add_callback(self, callback):
        """
        添加一个回调函数，注意，不能添加重复的 callback，这不推荐，也没有必要。
        :type callable,str
        :param callback:
        :return:
        """
        msg = None
        cb_name = callback.__class__.__name__
        if callback not in self._callback_set and cb_name in self._callback_name_set:
            msg = "Callback duplicate."
            callback.on_hook_failed(self, msg)

        if msg is not None:
            return False
        bisect.insort(self._callback_set, callback)
        self._callback_name_set.add(cb_name)

        callback._trainer = self
        callback.on_hooked(self, self.params)
        self.logger.info("{} hooked on {}.".format(callback, self))
        return True

    def reload_callback(self, callback):
        """重新加载某 callback"""
        self.remove_callback(callback.__class__)
        return self.add_callback(callback)

    def remove_callback(self, callback):
        """
        移除已加载的 callback
        Args:
            callback: 可以是回调类名、实例、或回调类类型

        Returns:
            是否移除成功，若返回False，则表明没有找到对应的 callback
            若返回 True，则表明该 callback 已被完好移除
        """
        msg = None
        from .callbacks import BaseCallback

        try:
            if issubclass(callback, BaseCallback):
                for cb in self._callback_set:
                    if cb.__class__.__name__ == callback.__name__:
                        callback = cb
                        break
        except:  # handle TypeError: issubclass() arg 1 must be a class
            pass

        if isinstance(callback, str):
            for cb in self._callback_set:
                if cb.__class__.__name__ == callback:
                    callback = cb
                    break

        if callback not in self._callback_set:
            return False

        cb_name = callback.__class__.__name__
        self._callback_set.remove(callback)
        self._callback_name_set.remove(cb_name)
        self.logger.info("{} unhooked from {}.".format(callback, self))
        return True

    def change_mode(self, train=True):
        for k, v in self._model_dict.items():
            if train:
                v.train()
            else:
                v.eval()

    def train(self):
        pass

    def train_epoch(self):
        pass

    def train_batch(self):
        pass

    def test(self):
        pass

    def evaluate(self):
        pass

    def predict(self, batch):
        """alias of inference"""
        self.inference(batch)

    def inference(self, batch):
        pass

    def inference_sample(self, sample):
        raise NotImplementedError()

    def callbacks(self, params: Params):
        """初始化回调函数"""
        pass

    def datasets(self, params: Params):
        """初始化数据集"""
        pass

    def models(self, params: Params):
        """初始化模型"""
        pass


class Trainer(BaseTrainer):

    def callbacks(self, params: Params):
        pass

    def datasets(self, params: Params):
        pass

    def models(self, params: Params):
        pass

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        pass

    def extra_state_dict(self) -> dict:
        return super().extra_state_dict()

    def load_extra_state_dict(self, state_dict, strict=False):
        super().load_extra_state_dict(state_dict, strict)


class WrapTrainer(Trainer):

    def __init__(self, params: Params, train_dataloader, model_with_loss_fn, optimize, eval_dataloader=None,
                 test_dataloader=None):
        super().__init__(params)

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)


class DistributedTrainer():
    def __init__(self, trainer_cls, params: Params, op):
        self.trainer_cls = trainer_cls
        self.params = params
        self.op = op

    def run(self):
        trainer = self.trainer_cls.__new__(self.trainer_cls)  # type:Trainer
        trainer._model_dict = {}  # type:Dict[str,torch.nn.Module]
        trainer._optim_dict = {}  # type:Dict[str,Optimizer]
        trainer._other_state_dict = {}
        trainer._vector_dict = {}
        trainer._checkpoint_plug = {}
        trainer._datasets_dict = {}  # type:Dict[str,DataBundler]
        trainer.train_epoch_toggle = False
        trainer.train_toggle = False
        trainer.experiment = None

        params = self.params
        if self.params is not None:
            trainer.params = params
            if isinstance(params.device, str):
                trainer.regist_device(torch.device(params.device))
            else:
                assert False

            if params.contains('tmp_dir'):
                if params.tmp_dir is not None:
                    os.environ['TMPDIR'] = params.tmp_dir

            if params.local_rank >= 1:
                from lumo import globs
                globs['rank'] = params.local_rank
        else:
            trainer.params = Params()

        from .experiment import Experiment
        # build experiment
        trainer.params.initial()

        if not trainer.params.get('git_commit', True):
            os.environ[_OS_ENV.THEXP_COMMIT_DISABLE] = '1'

        exp_name = _gene_class_exp_name(trainer)
        trainer.experiment = Experiment(exp_name)

        # rigist and save params of this training procedure
        trainer.experiment.add_params(params)

        # regist trainer info
        trainer_kwargs = _gene_trainer_info(trainer)
        trainer.experiment.add_plugin(_BUILTIN_PLUGIN.trainer, trainer_kwargs)

        params.distributed = True
        import torch.multiprocessing as mp
        if params.world_size == -1:
            params.world_size = torch.cuda.device_count()

        mp.spawn(mp_agent,
                 args=(trainer, self.op),
                 nprocs=trainer.params.world_size)
