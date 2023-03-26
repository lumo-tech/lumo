import bisect
import os
import warnings
from functools import lru_cache
from typing import Union, Dict, Any, Optional, Sequence, Mapping, Callable

import numpy as np
import torch
from .accelerator import get_accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lumo.utils.device import send_to_device
from lumo.core import TrainStage, Record, MetricType, Meter
from lumo.core.disk import TableRow, Metrics
from lumo.data import DataModule
from lumo.data.loader import DataLoaderType, DataLoaderSide
from lumo.proc import dist
from lumo.proc import glob
from lumo.utils import safe_io as IO
from lumo.trainer.rnd import RndManager
from lumo.utils.logger import Logger
from .backend.base import Accelerator
from .base import _BaseTrainer
from .components import TrainerExperiment, TrainerParams
from .saver import Saver

ParamsType = TrainerParams


class Trainer(_BaseTrainer):
    """
    Trainer provides a complete training and evaluation logic to help users focus on specific training details.
    It integrates logging, metric recording, experiment saving, version control, and complete callback control
    of each life cycle.

    When to use lumo.Trainer:
     - At the beginning of the investigation direction, there will be many code branches and experiments
        based on the same set of data and models, but with different training details.
     - Reproduce multiple papers in the same field, which are also based on the same dataset and evaluation logic.

    When to simply use lumo.SimpleExperiment:
     - You have maintained a mature training framework yourself, but there are still certain requirements
        for allocating dynamic(different) storage path and version control in each program.

    At present, I have explored best practices based on lumo in two fields, this includs:
     - https://github.com/pytorch-lumo/image-classification
     - https://github.com/pytorch-lumo/MMERC

    Both two reposiroties are github templates and you can use them to create you new (private) repository.
    """

    callback_function = {
        # "save_checkpoint", "save_model", "load_state_dict",
        'imodels',
        'train', 'test', 'evaluate',
        'train_epoch', 'train_step', 'test_step', 'evaluate_step',
        'process_loader', 'regist_dataloader'
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == 'Trainer':
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} directly, please create a subclass of it.")

    def __init__(self, params: ParamsType, dm: DataModule = None, accelerator=None):
        if dm is None:
            dm = DataModule(params)
        else:
            if dm.params is None:
                dm.params = params

        self.dm = dm  # type: DataModule
        self.params = params
        self._logger = None
        self._saver = None

        self.params.iparams()
        self.exp = TrainerExperiment(exp_name=self.generate_exp_name())

        self._database = TableRow(self.exp.mk_ipath('metric.pkl'), persistent=self.is_main)
        self.metric_board = Metrics(self.exp.mk_bpath('board.sqlite'), persistent=self.is_main)
        self.metric = self.exp.metric

        # self.exp.dump_info('table_row', self._database.fpath)
        self.rnd = RndManager()

        self.train_epoch_toggle = False
        self.train_toggle = False

        device = params.get('device', None) if not self.is_dist else None

        if isinstance(accelerator, str) or accelerator is None:
            accelerator = glob.get('accelerator', 'accelerator')
            accelerate = get_accelerator(accelerator)
        elif isinstance(accelerator, Accelerator):
            accelerate = accelerator
        else:
            raise NotImplementedError()

        self.accelerate = accelerate
        self.accelerate.set_device(torch.device(device))

        if dist.is_main():
            self.params.to_yaml(self.exp.params_fn)
            params_hash = self.params.hash()
            self.exp.dump_info('trainer', {
                'params_meta': {
                    'fn': self.exp.params_fn,
                    'hash': params_hash
                },
                'board_fn': self.metric_board.fpath
            }, append=True)
            self.exp.dump_info('params', self.params.to_dict())

        self.set_global_steps(0)
        self.set_epoch_idx(0)
        self.set_idx(0)
        if params.get('debug', False):
            self.exp.dump_info('debug', True)

    @property
    def metrics(self):
        return self.metric_board

    @property
    def db(self):
        return self._database

    @property
    def database(self):
        warnings.warn('TableRow is deprecated and will be removed soon, please use self.metric instead')
        return self._database

    @property
    def saver(self) -> Saver:
        if self._saver is None:
            self._saver = Saver(self.exp.state_dict_dir)
        return self._saver

    @property
    def is_debug(self):
        return self.params.get('debug', False)

    @property
    def is_main(self):
        return dist.is_main()

    @property
    def is_dist(self):
        return dist.is_dist()

    @property
    def local_rank(self):
        return dist.local_rank()

    @property
    def world_size(self):
        return dist.world_size()

    @property
    def logger(self):
        if self._logger is None:
            from lumo.utils.logger import set_global_logger
            self._logger = Logger()
            self._logger.use_stdout = glob.get('TRAINER_LOGGER_STDIO', True)

            set_global_logger(self._logger)
            if self.params.get('debug', False):
                self._logger.set_verbose(Logger.V_DEBUG)
                self._logger.debug('Enable debug log.')
            if self.is_main:
                fn = self._logger.add_log_dir(self.exp.log_dir)
                self.exp.dump_info('trainer', {
                    'logger_fn': fn
                }, append=True)

        return self._logger

    @property
    def safe_writer(self):
        """see trainer.writer"""
        try:
            import tensorflow as tf
            import tensorboard as tb
            tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        except ImportError:
            pass

        return self.writer

    @property
    @lru_cache()
    def writer(self):
        """
        Notes:
        ------
        When using add_embedding, there may raise some exceptions cased by version conflict, here is some solutions:

        1. `tensorflow_core._api.v1.io.gfile` or `tensorflow_core._api.v2.io.gfile` has no attribute `get_filesystem`
        first, try upgrade tensorboard and tensorflow as followed version:
            tensorboard==2.0.2
            tensorflow==2.0.0

        If you still have the same problem, use this code as a temporary solution:

            import tensorflow as tf
            import tensorboard as tb
            tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

        Use `trainer.safe_writer` to get a writter with these code added inside thexp.

        > This solution is referred by https://github.com/pytorch/pytorch/issues/30966 .


        2. You may cause PermissionError like: [Errno 13] Permission denied: '/tmp/.tensorboard-info/pid-20281.info'.
        The solution is to set environment variable TMPDIR

            export TMPDIR=/tmp/$USER;
            mkdir -p $TMPDIR;
            tensorboard --logdir ...

        One line command version:
            export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir ....

        > This solution is referred by https://github.com/tensorflow/tensorboard/issues/2010 .

        Returns:
            A SummaryWriter instance
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            try:
                from tensorboardX import SummaryWriter
            except ModuleNotFoundError:
                return None

        kwargs = self.exp.board_args
        res = SummaryWriter(**kwargs)

        def close(*args):
            """close writer"""
            res.flush()
            res.close()

        self.exp.add_exit_hook(close)
        return res

    @property
    def first_epoch(self):
        """Determine whether the current epoch is the first"""
        return self.eidx == 0

    @property
    def first_step(self):
        """Determine whether the current step is the first"""
        return self.global_steps == 0

    @property
    def idx(self):
        # started from 0
        return self._prop.get('idx', 0)

    @property
    def eidx(self):
        """
        This value will be automatically incremented by 1 before calling the train_epoch method
        current epoch round. The value remains unchanged in below methods:
         - `.on_train_epoch_begin()`: eidx
         - `.train_step()`: eidx
         - `.on_train_epoch_end()`: eidx

        This behaviour of `.idx` and `.global_steps` is the same as `.eidx`
        """
        return self._prop.get('eidx', 0)

    @property
    def global_steps(self) -> int:
        """started from 0"""
        return self._prop['global_steps']

    @property
    def trainer_state(self) -> Any:
        """
        Get the state of the Trainer object.

        Returns:
            Any: The state of the Trainer object.
        """
        return self._prop

    @property
    def devices(self) -> Dict[str, torch.device]:
        """
        Get the dictionary of devices used in the training session.

        Returns:
            Dict[str, torch.device]: A dictionary containing the devices used in the training session.
        """
        return {key: self[key] for key in self._state_dicts['devices']}

    @property
    def model_dict(self) -> Dict[str, nn.Module]:
        """
        Get the dictionary of model objects used in the training session.

        Returns:
            Dict[str, nn.Module]: A dictionary containing the model objects used in the training session.
        """
        return {key: self[key] for key in self._state_dicts['models']}

    @property
    def optim_dict(self) -> Dict[str, Optimizer]:
        """
        Get the dictionary of optimizer objects used in the training session.

        Returns:
            Dict[str, Optimizer]: A dictionary containing the optimizer objects used in the training session.
        """
        return {key: self[key] for key in self._state_dicts['optims']}

    @property
    def torch_tensor(self) -> Dict[str, torch.Tensor]:
        """
        Get the dictionary of PyTorch tensor objects used in the training session.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the PyTorch tensor objects used in the training session.
        """
        return {key: self[key] for key in self._state_dicts['tensor.th']}

    @property
    def numpy_tensor(self) -> Dict[str, np.ndarray]:
        """
        Get the dictionary of NumPy array objects used in the training session.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the NumPy array objects used in the training session.
        """
        return {key: self[key] for key in self._state_dicts['tensor.np']}

    @property
    def others(self) -> Dict[str, Any]:
        """
        A dictionary of additional attributes stored in the Trainer.

        Returns:
            Dict[str, Any]: The dictionary of additional attributes.
        """
        return {key: self[key] for key in self._state_dicts['others']}

    @property
    def datamodule(self) -> DataModule:
        """
        Returns the DataModule associated with this Trainer.

        Returns:
            DataModule: The DataModule associated with this Trainer.
        """
        return self.dm

    @property
    def train_dataloader(self) -> Optional[DataLoaderType]:
        """
        Returns the DataLoader for the training data.

        Returns:
            Optional[DataLoaderType]: The DataLoader for the training data, or None if it is not available.
        """
        return self.datamodule['train']

    @property
    def test_dataloader(self) -> Optional[DataLoaderType]:
        """
        Returns the DataLoader for the test data.

        Returns:
            Optional[DataLoaderType]: The DataLoader for the test data, or None if it is not available.
        """
        return self.datamodule['test']

    @property
    def val_dataloader(self) -> Optional[DataLoaderType]:
        """
        Returns the DataLoader for the validation data.

        Returns:
            Optional[DataLoaderType]: The DataLoader for the validation data, or None if it is not available.
        """
        return self.datamodule['val']

    @property
    def device(self):
        """
        Returns the device used for training.

        Returns:
            The device used for training.
        """
        return self.accelerate.device

    def _load_fun_state_dict(self, src: dict):
        """
        Loads state dicts into the Trainer's attributes.

        Args:
            src (dict): A dictionary of state dicts to be loaded.
        """
        for k, v in src.items():
            if self._rev_index.get(k, None) is not None:
                self[k].load_state_dict(v)

    def regist_dataloader(self, dataloader: DataLoader, stage: TrainStage):
        """
        Registers a dataloader with a given training stage to the current datamodule.

        Args:
            dataloader (DataLoader): The dataloader to be registered.
            stage (TrainStage): The training stage to which the dataloader will be associated.

        Returns:
            None
        """
        self.datamodule.regist_dataloader_with_stage(stage, dataloader)

    def process_loader(self, dm: Union[DataModule, DataLoader] = None, stage: TrainStage = TrainStage.train):
        """
        Prepares and registers a dataloader with the given training stage to the current datamodule.

        Args:
            dm (Union[DataModule, DataLoader], optional): The datamodule or dataloader to be processed. If not provided,
            the current datamodule will be used if it exists.
            stage (TrainStage, optional): The training stage to which the dataloader will be associated. Defaults to TrainStage.train.

        Returns:
            DataLoader: The prepared and registered dataloader.
            None: If the dataloader cannot be prepared or registered.
        """

        assert stage is not None, '`stage` cannot be None'
        if dm is None and self.dm is not None:
            dm = self.dm

        if isinstance(dm, DataModule):
            loader = dm[stage.value]
            if loader is None:
                # where datamodule.idataloader() methods first invoked (automaticly).
                loader = dm.get_loader_with_stage(stage)
                if loader is None:
                    return None
                loader = self.prepare_dataloader(loader, stage)
                self.regist_dataloader(loader, stage=stage)
        elif isinstance(dm, DataLoader):
            loader = dm
            loader = self.prepare_dataloader(loader, stage)
            self.regist_dataloader(loader, stage=stage)
        else:
            return None

        return loader

    def save_state_dict(self, name='latest.pth', dirpath=None, only_main=True):
        """
        Saves the current state dictionary to a file.

        Args:
            name: The name of the file to save the state dictionary to. Defaults to 'latest.pth'.
            dirpath: The directory path to save the state dictionary file to. If None, defaults to the state dictionary
                directory of the Trainer's experiment.
            only_main: If True, saves the state dictionary to a single file. If False and the Trainer is distributed,
                saves the state dictionary to multiple files, one for each process.

        Returns:
            The path to the saved state dictionary file.
        """
        if not only_main and self.is_dist:
            pre, ext = os.path.splitext(name)
            name = f'{pre}-{self.local_rank}{ext}'
        if dirpath is None:
            fn = os.path.join(self.exp.state_dict_dir, name)
        else:
            fn = os.path.join(dirpath, name)
        torch.save(self.state_dict(), fn)
        self.wait_for_everyone()
        return fn

    def load_state_dict(self, state_dict: dict):
        """Load state dictionary from a given dictionary.
        Args:
            state_dict (dict): A dictionary containing the state dictionary to be loaded.

        Returns:
            None
        """
        _sub = {'models', 'optims', 'other'}
        _missing = []

        for k, v in state_dict.items():
            if k in _sub:
                self._load_fun_state_dict(v)
            else:
                for kk, vv in v.items():
                    self[kk] = vv
        return

    def to_device(self, item: Optional[Union[nn.Module, torch.Tensor, Sequence, Mapping]] = None,
                  device: torch.device = None):
        """Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device."""
        if item is None:
            for k, v in list(self.model_dict.items()):
                self[k] = self.accelerate.prepare(v)
            for k, v in list(self.optim_dict.items()):
                self[k] = self.accelerate.prepare(v)
        else:
            if device is None:
                device = self.device
            item = send_to_device(item, device)
            return item

    def on_trainer_exception(self, func: Callable, exception: BaseException):
        """Updates database with error information when an exception occurs during training."""
        # self.exp.dump_info('exception', dict(end=strftime(),
        #                                      finished=False,
        #                                      error=str(exception),
        #                                      trainer_frame=str(func)), append=True)

    @property
    def is_initialized(self):
        """Whether this Trainer is initialized."""
        if self._prop.get('initial', False):
            return True
        return False

    def initialize(self):
        """
        Initializes the Trainer object, update meta information in Experiment and TableRow.

        If the Trainer object is already initialized, this method does nothing.

        This function is auto called when start train()/test()/evaluate()
        """

        if self.is_initialized:
            return
        self.exp.start()

        self.icallbacks(self.params)
        self.set_property('initial.callbacks', True)
        self.imodels(self.params)
        self.set_property('initial.model', True)
        self.set_property('initial', True)

        self.logger.info('Use Experiment')
        self.logger.info(self.exp)

    def stop_train(self):
        """Toggle to stop train."""
        self.train_toggle = True
        self.train_epoch_toggle = True

    def stop_train_epoch(self):
        """Toggle to skip current train epoch."""
        self.train_epoch_toggle = True

    def prepare_dataloader(self, loader: DataLoaderType, stage: TrainStage = None):
        """
        automatically called before train()/test()/evaluate(), see __new__ function of Trainer
        :param loader:
        :param stage:
        :return:
        """
        if isinstance(loader, DataLoader):
            loader = self.accelerate.prepare_data_loader(loader)
        elif isinstance(loader, DataLoaderSide):
            loader = loader.copy()
            loader._loaders = {k: self.prepare_dataloader(v, stage) for k, v in loader._loaders.items()}
        return loader

    def train(self, dm: Union[DataModule, DataLoaderType] = None, params: ParamsType = None, limit_global_steps=None):
        """Trains the model using the specified data loader and parameters.

        Args:
            dm (Union[DataModule, DataLoaderType], optional): The data loader or data module to use for training.
                Defaults to self.train_dataloader.
            params (ParamsType, optional): The training parameters to use. Defaults to None.
            limit_global_steps (int, optional): The maximum number of global steps to train for. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary of training results.

        Raises:
            ValueError: If no data loader is available for training.

        """
        self.change_stage(TrainStage.train)

        loader = self.select_loader(dm)
        if not loader:
            loader = self.train_dataloader

        if loader is None:
            self.set_property('early_stop', 'Lack of train loader')
            return self._prop

        if params is None:
            params = self.params

        for eidx in range(params.epoch):
            # update training progress
            self.set_epoch_idx(eidx)

            # train loop
            epoch_record = self.train_epoch(loader, params, limit_global_steps=limit_global_steps)

            # self.set_property('record', epoch_record)

            # early stop `train_toggle`
            if self.train_toggle:
                self.set_property('early_stop', 'train toggle')
                self.train_toggle = False
                break

            # early stop by `global_steps`
            if limit_global_steps is not None and self.global_steps >= limit_global_steps:
                self.set_property('early_stop', f'meet limit_global_steps {limit_global_steps}')
                break

            self.exp.dump_train_eidx(eidx, params.epoch)

        self.exp.end()
        return self._prop

    def train_epoch(self, loader: DataLoaderType, params: ParamsType = None,
                    limit_step=None,
                    limit_global_steps=None) -> Record:
        """Trains the model for one epoch using the specified data loader and parameters.

        Args:
            loader (DataLoaderType): The data loader to use for training.
            params (ParamsType, optional): The training parameters to use. Defaults to None.
            limit_step (int, optional): The maximum number of steps to train for. Defaults to None.
            limit_global_steps (int, optional): The maximum number of global steps to train for. Defaults to None.

        Returns:
            Record: A record of training results for the epoch.

        """
        stage = TrainStage.train
        self.change_stage(stage)
        record = self.create_record(stage=stage)

        if params is None:
            params = self.params

        self.wait_for_everyone()
        for idx, batch in enumerate(loader):
            batch = send_to_device(batch, device=self.device)
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

            if limit_step is not None and idx > limit_step:
                break
            if limit_global_steps is not None and self.global_steps >= limit_global_steps:
                break

            self.set_idx(idx)
            self._prop['global_steps'] += 1
            metric = self.train_step(batch, params)
            record.record(metric)

        record.flush()
        return record

    def set_property(self, key: str, value: any) -> None:
        """
        Sets a property with the given key to the given value.

        Args:
            key: A string representing the name of the property.
            value: The value to assign to the property.

        Returns:
            None
        """
        self._prop[key] = value

    def set_global_steps(self, val: int) -> None:
        """
        Sets the global step count to the given value.

        Args:
            val: An integer representing the global step count.

        Returns:
            None
        """
        self.set_property('global_steps', val)

    def set_epoch_idx(self, val: int) -> None:
        """
        Sets the current epoch index to the given value.

        Args:
            val: An integer representing the current epoch index.

        Returns:
            None
        """
        self.set_property('eidx', val)

    def set_idx(self, val: int) -> None:
        """
        Sets the current index to the given value.

        Args:
            val: An integer representing the current index.

        Returns:
            None
        """
        self.set_property('idx', val)

    @property
    def trainstage(self) -> TrainStage:
        return self._prop.get('stage', TrainStage.default)

    def set_stage(self, val: TrainStage):
        """
        Sets the training stage to the given value.

        Args:
            val (TrainStage): The value to set the training stage to.
        """
        self.set_property('stage', val)

    def add_callback(self, callback):
        """
        Adds a callback function. Note that duplicate callbacks are not recommended and not necessary.

        Args:
            callback: The callback function to add.

        Returns:
            bool: True if the callback was added successfully, False otherwise.
        """
        msg = None
        cb_name = callback.__class__.__name__
        cb_names = {cb.__class__.__name__ for cb in self.callbacks}
        if callback not in self.callbacks and cb_name in cb_names:
            msg = "Callback duplicate."
            callback.on_hook_failed(self, msg)
            return False

        if callback.only_main_process and not self.is_main:
            msg = f"{callback.__class__.__name__} only_main_process but in local_rank {self.local_rank}"
            callback.on_hook_failed(self, msg)
            return False

        if callback.only_single_gpu and not self.is_dist:
            msg = f"{callback.__class__.__name__} only_single_gpu but dist={self.is_dist}"
            callback.on_hook_failed(self, msg)
            return False

        if msg is not None:
            return False
        bisect.insort(self.callbacks, callback)

        callback._hooked = self
        callback.on_hooked(self, self.params)
        self.logger.info(f'{callback} hooked')
        return True

    def remove_callback(self, cur):
        """
        Removes the given callback from the list of callbacks.

        Args:
            cur: The callback to remove.
        """
        self.callbacks.remove(cur)
        pass

    def change_stage(self, stage: TrainStage):
        """
        Changes the training stage to the given value.

        Args:
            stage (TrainStage): The value to change the training stage to.
        """
        if self.trainstage == stage:
            return

        self.set_stage(stage)
        for k, v in self.model_dict.items():
            if 'ema' in k.lower():
                continue
            if stage.value:
                v.train()
            else:
                v.eval()

    def select_loader(self, dm=None, stage=None):
        """
        Selects the appropriate loader based on the given data module.

        Args:
            dm (DataModule or DataLoader or DataLoaderSide, optional): The data module to use. Defaults to None.

        Returns:
            DataLoader or None: The appropriate loader based on the given data module, or None if dm is None.
        """
        loader = None
        if dm:
            if isinstance(dm, DataModule):
                loader = dm.get_loader_with_stage(stage=self.trainstage)
                # loader = dm.train_dataloader
            elif isinstance(dm, DataLoader) or isinstance(dm, DataLoaderSide):
                loader = dm
            else:
                raise TypeError(type(dm))
        return loader

    def test(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step=None):
        """
        Tests the model on a given dataset and returns a `Record` object containing the evaluation results.

        Args:
            dm (Union[DataModule, DataLoader], optional): A `DataModule` or `DataLoader` object for the dataset to test on.
            params (ParamsType, optional): A dictionary containing hyperparameters for the test.
            limit_step (int, optional): An integer specifying the maximum number of batches to test on.
        Returns:
            A `Record` object containing the evaluation results.
        """
        stage = TrainStage.test
        self.change_stage(stage)

        loader = self.select_loader(dm)
        if not loader:
            loader = self.test_dataloader

        if loader is None:
            return None

        if params is None:
            params = self.params

        record = self.create_record(stage=stage)
        self.wait_for_everyone()
        for idx, batch in enumerate(loader):
            batch = send_to_device(batch, device=self.device)
            if limit_step is not None and idx >= limit_step:
                break
            self.set_idx(idx)
            metric = self.test_step(batch, params)
            record.record(metric)

        record.flush()
        return record

    def evaluate(self, dm: Union[DataModule, DataLoader] = None, params: ParamsType = None, limit_step: int = None):
        """
        Evaluates the model on a given dataset and returns a `Record` object containing the evaluation results.
        Args:
            dm (Union[DataModule, DataLoader], optional): A `DataModule` or `DataLoader` object for the dataset to evaluate on.
            params (ParamsType, optional): A dictionary containing hyperparameters for the evaluation.
            limit_step (int, optional): An integer specifying the maximum number of batches to evaluate on.
        Returns:
            A `Record` object containing the evaluation results.
        """
        stage = TrainStage.val
        self.change_stage(stage)

        loader = self.select_loader(dm)
        if not loader:
            loader = self.val_dataloader
        if loader is None:
            return None

        if params is None:
            params = self.params

        record = self.create_record(stage=stage)
        for idx, batch in enumerate(loader):
            batch = send_to_device(batch, device=self.device)
            if limit_step is not None and idx >= limit_step:
                break
            self.set_idx(idx)
            metric = self.evaluate_step(batch, params)
            record.record(metric)

        record.flush()
        return record

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        """
        Runs a single training step on a batch of data and returns a dictionary of training metrics.
        Args:
            batch: A batch of data to train on.
            params (ParamsType, optional): A dictionary containing hyperparameters for the training step.
        Returns:
            A dictionary of training metrics.
        """
        pass

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        """
        Runs a single testing step on a batch of data and returns a dictionary of evaluation metrics.
        Args:
            batch: A batch of data to test on.
            params (ParamsType, optional): A dictionary containing hyperparameters for the testing step.
        Returns:
            A dictionary of evaluation metrics.
        """
        pass

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        """
        Runs a single evaluation step on a batch of data and returns a dictionary of evaluation metrics.
        Args:
            batch: A batch of data to evaluate on.
            params (ParamsType, optional): A dictionary containing hyperparameters for the evaluation step.
        Returns:
            A dictionary of evaluation metrics.
        """
        pass

    def imodels(self, params: ParamsType):
        """Initialize model in here"""
        pass

    def icallbacks(self, params: ParamsType):
        """Initialize callbacks in here"""
        pass

    def inference(self, batch):
        """Perform inference on a batch of data."""
        raise NotImplementedError()

    def predict(self, batch):
        """Make a prediction on a batch of data."""
        raise NotImplementedError()

    def optim_state_dict(self, wrap=True):
        """Get a dictionary of the state of the optimizers."""
        res = {k: v.state_dict() for k, v in self.optim_dict.items()}
        if wrap:
            res = {'optim': res}
        return res

    def model_state_dict(self, wrap=True):
        """Get a dictionary of the state of the models."""
        res = {k: self.accelerate.unwrap_model(v).state_dict() for k, v in self.model_dict.items()}
        if wrap:
            res = {'model': res}
        return res

    def other_state_dict(self, wrap=True):
        """Get a dictionary of the state of the other objects."""
        res = {k: v.state_dict() for k, v in self.others.items()}
        if wrap:
            res = {'other': res}
        return res

    def state_dict(self):
        """Get a dictionary of the state of the object."""
        res = {
            'optims': self.optim_state_dict(wrap=False),
            'models': self.model_state_dict(wrap=False),
            'others': self.other_state_dict(wrap=False),
            'thtensor': self.torch_tensor,
            'nptensor': self.numpy_tensor,
            # 'devices': self.devices,
        }

        return res

    def Meter(self):
        """
        Returns a new instance of the Meter class.

        Returns:
            Meter: A new instance of the Meter class.
        """
        return Meter()

    def create_record(self, stage: TrainStage = None):
        """
        Creates a new Record object with the specified TrainStage.

        Args:
            stage (TrainStage, optional): The TrainStage to use for the new Record object. If not provided, the TrainStage
            from the Trainer object will be used.

        Returns:
            Record: A new Record object with the specified TrainStage.
        """
        if stage is None:
            stage = self.trainstage
        record = Record(stage=stage)
        return record

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point
        """
        self.accelerate.wait_for_everyone()

    def save_best_model(self):
        """
        Saves the best model checkpoint and metadata.

        If the current process is the main process, saves the best model checkpoint as 'best_model.ckpt'
        and its metadata as 'best_model.json'. If not, saves the checkpoint and metadata with the process rank
        appended to the filename, e.g., 'best_model-<rank>.ckpt' and 'best_model-<rank>.json'.

        The saved metadata includes the global training steps and the value of the experiment's best metric.

        Args:
            self: the Experiment object.

        Returns:
            None.
        """
        if self.is_main:
            file = self.exp.mk_bpath('models', 'best_model.ckpt')
            file_info = self.exp.mk_bpath('models', 'best_model.json')
        else:
            file = self.exp.mk_bpath('models', f'best_model-{self.local_rank}.ckpt')
            file_info = self.exp.mk_bpath('models', f'best_model-{self.local_rank}.json')

        torch.save(self.state_dict(), file)

        res = {'global_steps': self.global_steps, 'metric': self.exp.metric.value}
        IO.dump_json(IO.filter_unserializable_values(res), file_info)

        self.logger.info(f'saved best model at {file}')
        self.wait_for_everyone()

    def save_last_model(self):
        """
        Saves the last model checkpoint and metadata.

        If the current process is the main process, saves the last model checkpoint as 'last_model.ckpt'
        and its metadata as 'last_model.json'. If not, saves the checkpoint and metadata with the process rank
        appended to the filename, e.g., 'last_model-<rank>.ckpt' and 'last_model-<rank>.json'.

        The saved metadata includes the global training steps and the value of the experiment's best metric.

        Args:
            self: the Experiment object.

        Returns:
            None.
        """
        if self.is_main:
            file = self.exp.mk_bpath('models', 'last_model.ckpt')
            file_info = self.exp.mk_bpath('models', 'last_model.json')
        else:
            file = self.exp.mk_bpath('models', f'last_model-{self.local_rank}.ckpt')
            file_info = self.exp.mk_bpath('models', f'last_model-{self.local_rank}.json')

        torch.save(self.state_dict(), file)

        res = {'global_steps': self.global_steps, 'metric': self.exp.metric.current}
        IO.dump_json(IO.filter_unserializable_values(res), file_info)

        self.logger.info(f'saved last model at {file}')
        self.wait_for_everyone()
