"""
"""

import inspect
import os
import tempfile
import time
from functools import wraps
from typing import TYPE_CHECKING, NewType, Any, Optional, Dict, Union

from torch.utils.data import DataLoader

from lumo.core import ParamsType, Meter, MetricType, Record, TrainStage, wrap_result
from lumo.data import DataModule
from lumo.utils.screen import inlinetqdm
from lumo.data.loader import summarize_loader, DataLoaderType
from lumo.utils import fmt

if TYPE_CHECKING:
    from .trainer import Trainer

_func_map = {
    'evaluate': 'eval',
    'evaluate_step': 'eval_step'
}


def map_func_name(name):
    return _func_map.get(name, name)


class BaseCallback:
    """
    Abstract base class used to build new callbacks.

    Callbacks are created to hook into the various stages of lumo's 'Trainer' training, inference or data preparing.
    You can hook any subclasses of `BaseCallback` in `lumo.kit.trainer.'Trainer'` class.

    To create a custom callback class, subclass `lumo.kit.callbacks.BaseCallback`, and define the callback methods
    you want to do something before or after that.

    The callback method must be named like on_xxx_begin()/on_xxx_end(), where `xxx` is a function in trainer.


    TrainCallback, SaveLoadCallback and InitialCallback define the most methods in 'Trainer', you can simply subclass these
    callback class to custom your callback.

    Notes:
        There is no need for you to call the on_xxx function in your function's logic.

        The 'Trainer' has some magic feature... not like keras or other framework with trainer/callback logic that
        you may need to manully call the callback function `on_xxx...()` in you overrided or newly defined method in 'Trainer',
        If you define a function `foo()` in your trainer, the callback function on_foo_begin() and on_foo_end() will be found
        automatically and be called, There is no need for you to call them in your function's logic.


    Examples:
        {{examples/docs/basecallback.py}}

        {{examples/docs/basecallback2.py}}

    """
    priority = 0  # type: int # All callbacks in lumo will have priority in range 0-100
    only_single_gpu = False  # Callback only works when single gpu mode
    only_main_process = False  # Callback only works when in main process

    def __new__(cls, *_, **__):
        self = super().__new__(cls)
        self._hooked = None  # type: Trainer

        def ecp_wrap(func):
            """Wrapper for cache exceptions"""

            @wraps(func)
            def on_exception(hooked: 'Trainer', tfunc, params: ParamsType, e: BaseException, *args, **kwargs):
                self.ecp = getattr(self, "ecp", None)
                res = None
                if self.ecp != e:
                    res = self.on_first_exception(hooked, tfunc, params, e, *args, **kwargs)
                    self.ecp = e

                eres = func(hooked, tfunc, params, e, *args, **kwargs)
                if res is None:
                    return eres
                else:
                    return res

            return on_exception

        self.on_exception = ecp_wrap(self.on_exception)
        return self

    def getfuncargs(self, func, *args, **kwargs):
        res = inspect.getcallargs(func, *args, **kwargs)
        if 'self' in res:
            res.pop('self')
        return res

    def on_hooked(self, source: 'Trainer', params: ParamsType):
        """called when callback hooked trainer"""
        pass

    def on_first_exception(self, source: 'Trainer', func, params: ParamsType, e: BaseException, *args, **kwargs):
        """
        when an exception was raised in some function, on_exception() will be called.

        如果异常发生在一个嵌套调用的函数中，那么该异常会在每一层函数都raise一次。

        该方法将被调用当该异常第一次raise出来的时候。
        该方法在 __new__ 中作了处理逻辑，不受继承关系影响
        """
        pass

    def on_exception(self, source: 'Trainer', func, params: ParamsType, e: BaseException, *args, **kwargs):
        """called when exception raised in some function"""
        return False

    def on_hook_failed(self, source, message):
        """Any reason when callback cannot hook on trainer"""
        pass

    def on_begin(self, source: 'Trainer', func, params: ParamsType, *args, **kwargs):
        """called before trainer.func is called"""
        cb_func = getattr(self, f"on_{map_func_name(func.__name__)}_begin", None)
        if cb_func is not None:
            cb_func(source, func, params, *args, **kwargs)

    def on_end(self, source: 'Trainer', func, params: ParamsType, result, *args, **kwargs):
        cb_func = getattr(self, f"on_{map_func_name(func.__name__)}_end", None)
        if cb_func is not None:
            cb_func(source, func, params, result, *args, **kwargs)

    def __le__(self, other):
        return self.priority <= other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def hook(self, source: 'Trainer'):
        source.add_callback(self)

    def unhook(self):
        self._hooked.remove_callback(self)

    def _repr_by_val(self, *vals):
        vstr = "; ".join(["{}={}".format(val, str(getattr(self, val, None))) for val in vals])
        return "<{}([{}]) at 0x{:016X}>".format(self.__class__.__name__, vstr, id(self))

    def __repr__(self) -> str:
        return self._repr_by_val("priority")


class TrainCallback(BaseCallback):
    """
    Abstract base class used to build new callbacks. Defined some methods used in train/eval/test loop.

    Subclass TrainCallback if you want to control or add behavior before or after
        - train/train_epoch/train_step
        - eval/eval_epoch/eval_step
        - test/test_epoch/test_step
        - predict
        - inference
    """

    def on_train_begin(self, trainer: 'Trainer', func, params: ParamsType,
                       dm: Union[DataModule, DataLoaderType] = None, arg_params: ParamsType = None,
                       *args, **kwargs):
        pass

    def on_train_epoch_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_test_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_eval_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_train_step_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_eval_step_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_test_step_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_train_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        pass

    def on_train_epoch_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args,
                           **kwargs):
        pass

    def on_test_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record = None, *args, **kwargs):
        pass

    def on_eval_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record = None, *args, **kwargs):
        pass

    def on_train_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType = None, *args,
                          **kwargs):
        pass

    def on_eval_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType = None, *args,
                         **kwargs):
        pass

    def on_test_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType = None, *args,
                         **kwargs):
        pass

    def on_predict_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_predict_end(self, trainer: 'Trainer', func, params: ParamsType, result: Any = None, *args, **kwargs):
        pass

    def on_inference_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_inference_end(self, trainer: 'Trainer', func, params: ParamsType, result: Any, *args, **kwargs):
        pass


class InitialCallback(BaseCallback):
    """
    Abstract base class used to build new callbacks. Defined some methods used in save/load.

    Subclass InitialCallback if you want to control or add behavior before or after
        - initialization of optims/models/dataloader
    """

    def on_imodels_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_imodels_end(self, trainer: 'Trainer', func, params: ParamsType, result: Any, *args, **kwargs):
        pass

    def on_prepare_dataloader_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        pass

    def on_prepare_dataloader_end(self, trainer: 'Trainer', func, params: ParamsType, loader: DataLoader,
                                  *args, **kwargs):
        pass

    def on_process_loader_begin(self, trainer: 'Trainer', func, params: ParamsType,
                                dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        pass

    def on_process_loader_end(self, trainer: 'Trainer', func, params: ParamsType, loader: DataLoader,
                              dm: DataModule, stage: TrainStage,
                              *args, **kwargs):
        pass

    def on_regist_dataloader_begin(self, trainer: 'Trainer', func, params: ParamsType,
                                   dataloader: DataLoader, stage: TrainStage, *args, **kwargs):
        pass

    def on_regist_dataloader_end(self, trainer: 'Trainer', func, params: ParamsType, result: Any,
                                 dataloader: DataLoader, stage: TrainStage, *args, **kwargs):
        pass


class EvalCallback(TrainCallback):
    """
    Callback for evaluation.

    Args:
        eval_per_epoch:
        test_per_epoch:
    """
    only_main_process = True

    def __init__(self, eval_per_epoch=1, test_per_epoch=10):

        self.eval_in_per_epoch = eval_per_epoch
        self.test_in_per_epoch = test_per_epoch

        # evaluate/test on train end
        self._last_eval = -1
        self._last_test = -1

    @property
    def has_eval(self):
        return self.eval_in_per_epoch is not None and self.eval_in_per_epoch > 0

    @property
    def has_test(self):
        return self.test_in_per_epoch is not None and self.test_in_per_epoch > 0

    def _test_or_eval(self, params: ParamsType, trainer: 'Trainer'):
        if self.has_eval:
            if params.eidx % self.eval_in_per_epoch == self.eval_in_per_epoch - 1:
                self._last_eval = params.eidx
                trainer.evaluate()
        if self.has_test:
            if params.eidx % self.test_in_per_epoch == self.test_in_per_epoch - 1:
                self._last_test = params.eidx
                trainer.test()

    def on_train_epoch_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args,
                           **kwargs):
        self._test_or_eval(params, trainer)

    def on_train_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        if self._last_eval != params.eidx and self.has_eval:
            trainer.evaluate()
        if self._last_test != params.eidx and self.has_test:
            trainer.test()

    def __repr__(self):
        return self._repr_by_val("eval_in_per_epoch", "test_in_per_epoch")


class DebugCallback(BaseCallback):

    def on_hooked(self, source: 'Trainer', params: ParamsType):
        super().on_hooked(source, params)
        print('on_hooked()')

    def on_first_exception(self, source: 'Trainer', func, params: ParamsType, e: BaseException, *args, **kwargs):
        super().on_first_exception(source, func, params, e, *args, **kwargs)
        print('on_first_exception()', func.__name__)
        # from rich.console import Console
        # console = Console()
        # try:
        #     import six
        #     six.reraise(type(e), e, e.__traceback__)
        #
        # except:

    def on_exception(self, source: 'Trainer', func, params: ParamsType, e: BaseException, *args, **kwargs):
        return super().on_exception(source, func, params, e, *args, **kwargs)

    def on_hook_failed(self, source, message):
        super().on_hook_failed(source, message)
        print('on_hook_failed()', message)

    def on_begin(self, source: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_begin(source, func, params, *args, **kwargs)
        print('on_begin()', func.__name__)

    def on_end(self, source: 'Trainer', func, params: ParamsType, result, *args, **kwargs):
        super().on_end(source, func, params, result, *args, **kwargs)
        print('on_end()', func.__name__)


class LoggerCallback(TrainCallback, InitialCallback):

    def __init__(self, step_frequence=3, break_in=1000):
        self.stage = {}
        self.breakin = break_in
        self.c = 0
        self.step = step_frequence
        file = tempfile.TemporaryFile('w')
        self.temp = file

    def on_imodels_end(self, trainer: 'Trainer', func, params: ParamsType, result: Any, *args, **kwargs):
        super().on_imodels_end(trainer, func, params, result, *args, **kwargs)
        trainer.logger.info('Model initialized.')

    def on_hooked(self, source: 'Trainer', params: ParamsType):
        super().on_hooked(source, params)
        source.logger.raw(params)

    def on_process_loader_end(self, trainer: 'Trainer', func, params: ParamsType, loader: DataLoaderType,
                              dm: DataModule,
                              stage: TrainStage, *args, **kwargs):
        super().on_process_loader_end(trainer, func, params, loader, dm, stage, *args, **kwargs)
        if loader is None:
            return
        if stage in self.stage:
            return
        loader_str = summarize_loader(loader)
        try:
            lsize = len(loader)
        except:
            lsize = None
        self.stage[stage] = lsize
        trainer.logger.info(f'{loader_str} for {stage.value} prepared.')

    def on_train_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_train_begin(trainer, func, params, *args, **kwargs)
        trainer.logger.info('[[Train Begin]]')
        self.global_tqdm = inlinetqdm(total=params.epoch,
                                      position=0, leave=True,
                                      bar_format='Ela: {elapsed} | Last: {remaining} | Avg: {rate}',
                                      file=self.temp)

    def renew(self, stage):
        """创建一个新的"""
        self.cur_tqdm = inlinetqdm(total=self.stage[stage], position=0, leave=True,
                                   bar_format='{desc}{elapsed}<{remaining} ({percentage:3.0f}%){postfix}',
                                   file=self.temp)
        self.record = Record()

    def update(self, trainer: 'Trainer'):
        self.c += 1
        self.cur_tqdm.update()
        if self.c % self.step == 0:
            trainer.logger.raw(self.cur_tqdm, inline=True)

        if self.c % self.breakin == 0 or ((trainer.idx + 1) == self.stage[TrainStage.train]):
            trainer.logger.inline(self.cur_tqdm)
            trainer.logger.newline()

    def flush(self, trainer: 'Trainer'):
        self.c = 0
        trainer.logger.inline(self.cur_tqdm)
        trainer.logger.newline()

    def on_train_epoch_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_train_epoch_begin(trainer, func, params, *args, **kwargs)
        self.renew(TrainStage.train)
        self.time = time.time()

    def on_test_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_test_begin(trainer, func, params, *args, **kwargs)
        self.renew(TrainStage.test)
        trainer.logger.info('[[Test Begin]]')

    def on_eval_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_eval_begin(trainer, func, params, *args, **kwargs)
        self.renew(TrainStage.val)
        trainer.logger.info('[[Evaluate Begin]]')

    @staticmethod
    def format_interval(t: float):
        """
        Formats a number of seconds as a clock time, [H:]MM:SS

        Parameters
        ----------
        t  : int
            Number of seconds.

        Returns
        -------
        out  : str
            [H:]MM:SS
        """
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        if h:
            return '{0:d}h{1:02d}m{2:02d}s'.format(h, m, s)
        else:
            return '{0:02d}m{1:02d}s'.format(m, s)

    def format_train_epoch_time(self, n, total, elapsed, ncols=None, prefix='', ascii=False, unit='it',
                                unit_scale=False, rate=None, bar_format=None, postfix=None,
                                unit_divisor=1000, initial=0, colour=None, **extra_kwargs):
        elapsed_str = self.format_interval(elapsed)
        remaining = (total - n) / rate if rate and total else 0
        remaining_str = self.format_interval(remaining) if rate else '?'

        rate_str = self.format_interval(1 / rate)

        last = time.time() - self.time
        last_str = self.format_interval(last)

        return f'{n}/{total} train: {elapsed_str} | epoch: {last_str} | last: {remaining_str} | avg: {rate_str}'

    def on_train_epoch_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, record, *args, **kwargs)
        self.global_tqdm.update()
        # self.flush(trainer)
        trainer.logger.info(self.format_train_epoch_time(**self.global_tqdm.format_dict))

    def on_test_end(self, trainer: 'Trainer', func, params: ParamsType, record: Optional[Record], *args, **kwargs):
        self.flush(trainer)
        trainer.logger.info('[[Test End]]')

    def on_eval_end(self, trainer: 'Trainer', func, params: ParamsType, record: Optional[Record], *args, **kwargs):
        self.flush(trainer)
        trainer.logger.info('[[Evaluate End]]')

    def on_train_end(self, trainer: 'Trainer', func, params: ParamsType, record: Optional[Record], *args, **kwargs):
        self.flush(trainer)
        trainer.logger.info('[[Train End]]')

    def on_train_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType, *args, **kwargs):
        self.record.record(metric)
        self.cur_tqdm.set_description_str(f"{trainer.idx + 1}/{self.stage[TrainStage.train]}, ", refresh=False)
        self.cur_tqdm.set_postfix(**self.record.avg(), refresh=False)
        self.update(trainer)

    def on_eval_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType, *args, **kwargs):
        self.record.record(metric)
        self.cur_tqdm.set_description_str(f"{trainer.idx + 1}/{self.stage[TrainStage.val]}, ", refresh=False)
        self.cur_tqdm.set_postfix(**self.record.avg(), refresh=False)
        self.update(trainer)

    def on_test_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType, *args, **kwargs):
        self.record.record(metric)
        self.cur_tqdm.set_description_str(f"{trainer.idx + 1}/{self.stage[TrainStage.test]}, ", refresh=False)
        self.cur_tqdm.set_postfix(**self.record.avg(), refresh=False)
        self.update(trainer)


class EpochCheckpoint(TrainCallback):
    """
    在 'Trainer' 训练过程中定时保存模型
    """
    only_main_process = True

    def __init__(self, per_epoch=50):
        self.per_epoch = per_epoch

    def on_train_epoch_end(self, trainer: 'Trainer', func, params: ParamsType, record: Optional[Record], *args,
                           **kwargs):
        meter = record.avg()
        if params.eidx % self.per_epoch == 0 and params.eidx > 0:
            trainer.save_checkpoint(meta_info=Meter.wrap_result(meter))

    def __repr__(self) -> str:
        return self._repr_by_val("per_epoch")


class GlobalStepCheckpoint(TrainCallback):
    only_main_process = True

    def __init__(self, per_step=2500):
        self.per = per_step

    def on_train_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: Meter, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, metric, *args, **kwargs)
        if params.global_step % self.per == 0 and params.global_step > 0:
            trainer.save_checkpoint(meta_info=Meter.wrap_result(metric))


class KeyErrorSave(TrainCallback):
    """
    Callback to save checkpoints when you interrupt the program.
    """
    only_main_process = True
    only_single_gpu = True
    priority = -1

    def __init__(self, wait_input=False):
        self.wait_input = wait_input

    def on_first_exception(self, source: 'Trainer', func, params: ParamsType, e: BaseException, *args, **kwargs):
        if isinstance(e, KeyboardInterrupt):
            source.logger.info("KeyErrorSave trigged, save checkpoint")
            source.save_checkpoint({"mode": "KeyboardInterrupt"})

            tp = "n"
            if self.wait_input:
                tp = input("continue train step? (y/other)")

            if tp.lower() == "y":
                return True


class EMAUpdate(TrainCallback):
    """
    Callback to update EMA model every train step.

    Variable in 'Trainer' instance is identified as EMA model when
     - is instance of torch.nn.Module
     - name is started with 'ema'
    """

    def on_train_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, metric, *args, **kwargs)
        for k, v in trainer.model_dict.items():
            if k.lower().startswith('ema'):
                v.step()


class AutoLoadModel(InitialCallback):
    """
    Callback to automatically load pretrained model.

    The load function will be executed if and only if
     - both the params `pretrain` and `pretrain_path` are defined in ParamsType
     - `pretrain` is True and `pretrain_path` is not None.
    """

    def on_imodels_end(self, trainer: 'Trainer', func, params: ParamsType, result: Any, *args, **kwargs):
        super().on_imodels_end(trainer, func, params, result, *args, **kwargs)
        if params.get('pretrain', False):
            path = params.get('pretrain_path', None)
            if path is not None:
                trainer.load_state_dict(path)


class EvalFirst(AutoLoadModel):
    """
    Callback to evaluation before the first train step. Aften used to debug or evaluate some dataset by restored model.
    """

    def __init__(self, datamodule=None):
        super().__init__()
        self.datamodule = datamodule

    def on_imodels_end(self, trainer: 'Trainer', func, params: ParamsType, result: Meter, *args, **kwargs):
        if params.get('eval_first', True):
            if self.datamodule is not None:
                trainer.evaluate(self.datamodule)
            else:
                trainer.evaluate()


class RecordCallback(TrainCallback):
    def log(self, metrics: MetricType, step, namespace):
        metrics = wrap_result(metrics)
        scalar_metrics = {}
        matrix_metrics = {}
        text_metrics = {}
        for k, v in metrics.items():
            v = fmt.to_ndarray(v)
            dtype = v.dtype.name
            elnum = v.size
            if 'str' in dtype:
                text_metrics[k] = str(v.tolist())
            else:
                if elnum == 1:
                    scalar_metrics[k] = v
                elif elnum > 1:
                    matrix_metrics[k] = v
        self.log_text(text_metrics, step, namespace)
        self.log_scalars(scalar_metrics, step, namespace)
        self.log_matrix(matrix_metrics, step, namespace)

    def log_text(self, metrics: Dict, step: int, namespace: str):
        return NotImplemented

    def log_scalars(self, metrics: Dict, step: int, namespace: str):
        return NotImplemented

    def log_matrix(self, metrics: Dict, step: int, namespace: str):
        return NotImplemented

    def on_hooked(self, source: 'Trainer', params: ParamsType):
        super().on_hooked(source, params)
        source.exp.set_prop('AutoRecord', self.__class__.__name__)

    def on_train_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, metric, *args, **kwargs)
        self.log(metric, step=trainer.global_steps, namespace='train.step')

    def on_train_epoch_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, record, *args, **kwargs)
        self.log(record.avg(), step=trainer.global_steps, namespace='train.epoch')

    def on_test_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        self.log(record.avg(), step=trainer.global_steps, namespace='test')

    def on_eval_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_eval_end(trainer, func, params, record, *args, **kwargs)
        self.log(record.avg(), step=trainer.global_steps, namespace='evaluate')


class WandbCallback(RecordCallback):

    def log(self, metrics: MetricType, step, namespace):
        metrics = {
            f"{namespace}.{k}": v
            for k, v in wrap_result(metrics).items()}
        self._hooked.wandb.log(metrics, step=step)

    def log_text(self, metrics: Dict, step: int, namespace: str):
        wandb = self._hooked.wandb
        metrics = {k: v for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    def log_scalars(self, metrics: Dict, step: int, namespace: str):
        wandb = self._hooked.wandb
        metrics = {k: wandb.Html(v) for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    def log_matrix(self, metrics: Dict, step: int, namespace: str):
        wandb = self._hooked.wandb
        metrics = {k: wandb.Image(v) for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    def on_first_exception(self, source: 'Trainer', func, params: ParamsType, e: BaseException, *args, **kwargs):
        super().on_first_exception(source, func, params, e, *args, **kwargs)


class TensorBoardCallback(RecordCallback):
    def log(self, metrics: MetricType, step, namespace):
        metrics = wrap_result(metrics)
        self._hooked.safe_writer.add_scalars(main_tag=namespace, tag_scalar_dict=metrics, global_step=step)

    def log_text(self, metrics: Dict, step: int, namespace: str):
        writer = self._hooked.safe_writer
        for k, v in metrics.items():
            writer.add_text(k, v, global_step=step)

    def log_scalars(self, metrics: Dict, step: int, namespace: str):
        writer = self._hooked.safe_writer
        writer.add_scalars(main_tag=namespace, tag_scalar_dict=metrics, global_step=step)

    def log_matrix(self, metrics: Dict, step: int, namespace: str):
        return NotImplemented
        # writer = self._hooked.safe_writer
        #
        # for k, v in metrics.items():
        #     writer


class StopByCode(TrainCallback):
    def __init__(self, step=100):
        self.step = step

    def on_train_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: Meter, *args, **kwargs):
        if trainer.global_steps % self.step == 0:
            if os.path.exists(trainer.exp.test_file('.stop')):
                trainer.exp.add_tag('lumo.early_stop')
                trainer.logger.info('Early stop this test manully.')
                trainer.stop_train_epoch()
                trainer.stop_train()


CallbackType = NewType('CallbackType', BaseCallback)
