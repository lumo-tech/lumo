"""
"""

import inspect
import os
import sys
from functools import wraps

from lumo.base_classes.trickitems import NoneItem, AvgItem
from lumo.proc.dist import is_main
from lumo.utils.timer import format_second, Timer
from .meter import AvgMeter
from .meter import Meter
from .mixin import CallbackMix
from .params import Params
from .trainer import Trainer, TrainerResult

_func_map = {
    'evaluate': 'eval',
    'evaluate_step': 'eval_step'
}


def map_func_name(name):
    return _func_map.get(name, name)


class BaseCallback:
    """
    Abstract base class used to build new callbacks.

    Callbacks are created to hook into the various stages of lumo's Trainer training, inference or data preparing.
    You can hook any subclasses of `BaseCallback` in `lumo.kit.trainer.Trainer` class.

    To create a custom callback class, subclass `lumo.kit.callbacks.BaseCallback`, and define the callback methods
    you want to do something before or after that.

    The callback method must be named like on_xxx_begin()/on_xxx_end(), where `xxx` is a function in trainer.


    TrainCallback, SaveLoadCallback and InitialCallback define the most methods in Trainer, you can simply subclass these
    callback class to custom your callback.

    Notes:
        There is no need for you to call the on_xxx function in your function's logic.

        The Trainer has some magic feature... not like keras or other framework with trainer/callback logic that
        you may need to manully call the callback function `on_xxx...()` in you overrided or newly defined method in Trainer,
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
        self._hooked = None

        def ecp_wrap(func):
            """Wrapper for cache exceptions"""

            @wraps(func)
            def on_exception(hooked: Trainer, tfunc, params: Params, e: BaseException, *args, **kwargs):
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

    def on_hooked(self, source: Trainer, params: Params):
        """called when callback hooked trainer"""
        pass

    def on_first_exception(self, source: Trainer, func, params: Params, e: BaseException, *args, **kwargs):
        """
        when an exception was raised in some function, on_exception() will be called.

        如果异常发生在一个嵌套调用的函数中，那么该异常会在每一层函数都raise一次。

        该方法将被调用当该异常第一次raise出来的时候。
        该方法在 __new__ 中作了处理逻辑，不受继承关系影响
        """
        pass

    def on_exception(self, source: Trainer, func, params: Params, e: BaseException, *args, **kwargs):
        """called when exception raised in some function"""
        return False

    def on_hook_failed(self, source, message):
        """Any reason when callback cannot hook on trainer"""
        pass

    def on_begin(self, source: Trainer, func, params: Params, *args, **kwargs):
        """called before trainer.func is called"""
        cb_func = getattr(self, f"on_{map_func_name(func.__name__)}_begin", None)
        if cb_func is not None:
            cb_func(source, func, params, *args, **kwargs)

    def on_end(self, source: Trainer, func, params: Params, result, *args, **kwargs):
        cb_func = getattr(self, f"on_{map_func_name(func.__name__)}_end", None)
        if cb_func is not None:
            cb_func(source, func, params, result, *args, **kwargs)

    def __le__(self, other):
        return self.priority <= other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def hook(self, source: CallbackMix):
        source.reload_callback(self)

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

    def on_train_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_train_epoch_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_test_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_eval_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_train_step_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_eval_step_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_test_step_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_train_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        pass

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        pass

    def on_test_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        pass

    def on_eval_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        pass

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_eval_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_test_step_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        pass

    def on_predict_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_predict_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_inference_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_inference_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass


class SaveLoadCallback(BaseCallback):
    """
    Abstract base class used to build new callbacks. Defined some methods used in save/load.

    Subclass SaveLoadCallback if you want to control or add behavior before or after
        - save_xxx
        - load_xxx
    """

    def on_save_keypoint_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_save_keypoint_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_save_model_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_save_model_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_save_checkpoint_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_save_checkpoint_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_load_state_dict_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_load_state_dict_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass


class InitialCallback(BaseCallback):
    """
    Abstract base class used to build new callbacks. Defined some methods used in save/load.

    Subclass InitialCallback if you want to control or add behavior before or after
        - initialization of optims/models/dataloader
    """

    def on_ioptims_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_ioptims_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_imodels_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_imodels_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        pass

    def on_prepare_dataloader_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        pass

    def on_prepare_dataloader_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
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

    def _test_or_eval(self, params: Params, trainer: Trainer):
        if self.has_eval:
            if params.eidx % self.eval_in_per_epoch == self.eval_in_per_epoch - 1:
                self._last_eval = params.eidx
                trainer.evaluate()
        if self.has_test:
            if params.eidx % self.test_in_per_epoch == self.test_in_per_epoch - 1:
                self._last_test = params.eidx
                trainer.test()

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        self._test_or_eval(params, trainer)

    def on_train_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        if self._last_eval != params.eidx and self.has_eval:
            trainer.evaluate()
        if self._last_test != params.eidx and self.has_test:
            trainer.test()

    def __repr__(self):
        return self._repr_by_val("eval_in_per_epoch", "test_in_per_epoch")


class LoggerCallback(TrainCallback, InitialCallback, SaveLoadCallback):
    """
    Callback to log info produced during whole Trainer lifecycle.
    """
    # only_main_process = True
    priority = 100

    def __init__(self, avg=True, step_frequence=3, breakline_in=1000):
        self.avg = avg
        self.step_frequence = step_frequence
        self.breakline = breakline_in
        self.history_loader = set()
        self.is_main = is_main()

    @property
    def meter(self):
        if self._meter is None:
            self.reset_meter()
        return self._meter

    def reset_meter(self):
        if self.avg:
            meter = AvgMeter()
        else:
            meter = Meter()
        self._meter = meter

    def _need_log(self, step):
        if not self.is_main:
            return False
        return self.step_frequence > 0 and step % self.step_frequence == 0

    def _need_breakline(self, step):
        if not self.is_main:
            return False
        return self.breakline > 0 and step % self.breakline == 0

    def on_hooked(self, source: Trainer, params: Params):
        source.logger.raw(' '.join(sys.argv))
        source.logger.info("Exp BaseDir", os.path.abspath(source.exp.exp_root))
        source.logger.info("Exp Trainer", source.__class__.__name__)
        source.logger.info("Exp TestDir", source.exp.test_root)
        source.logger.raw(params)
        self.start = 0
        self.cur = None
        self._meter = None

    def on_train_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.start = params.eidx
        self.traintime = Timer()
        self.traintime.start()
        trainer.logger.info('[[Train Begin]]')
        super().on_train_begin(trainer, func, params, *args, **kwargs)

    def on_train_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        self.traintime.end()
        meter = result.meter
        if meter is None:
            meter = ""
        trainer.logger.info(f"[[Train End in {format_second(self.traintime['use'])}]]", meter)

    def on_train_epoch_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.reset_meter()
        self.epochtime = Timer()
        self.epochtime.start()
        trainer.logger.info("{}/{}".format(params.eidx, params.epoch))

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        self.traintime.mark("epoch")
        self.epochtime.end()
        if self.cur is None:
            self.cur = params.eidx

        avg = self.traintime["use"] / (self.cur - self.start + 1)
        self.cur += 1
        last = (params.epoch - params.eidx) * avg

        tm = Meter()
        tm.train = format_second(self.traintime["use"])
        tm.epoch = format_second(self.epochtime["use"])
        tm.avg = format_second(avg)
        tm.last = format_second(last)
        # trainer.logger.newline()
        trainer.logger.info(tm)

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        meter = Meter.wrap_result(meter)
        self.meter.update(meter)
        meter = self.meter
        if self._need_breakline(params.idx):
            trainer.logger.newline()
            self.reset_meter()
        if self._need_log(params.idx):
            trainer.logger.inline("{}/{}".format(params.idx + 1, len(trainer.train_dataloader)), meter, fix=1)

    def on_first_exception(self, source: Trainer, func, params: Params, e: BaseException, *args, **kwargs):
        source.logger.error("{} raised".format(e.__class__.__name__))

    def on_test_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        trainer.logger.newline()
        self.reset_meter()

    def on_test_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        if meter is None:
            meter = ""
        trainer.logger.info("[[Test]]", meter)
        self.reset_meter()

    def on_test_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        meter = Meter.wrap_result(meter)
        self.meter.update(meter)
        meter = self.meter
        trainer.logger.inline("[[Test]]", "{}/{}".format(params.idx + 1, len(trainer.test_dataloader)), meter, fix=1)

    def on_eval_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        trainer.logger.newline()
        self.reset_meter()

    def on_eval_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        if meter is None:
            meter = ""
        trainer.logger.info("[[Eval]]", meter)

    def on_eval_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        meter = Meter.wrap_result(meter)
        self.meter.update(meter)
        meter = self.meter
        trainer.logger.inline("[[Eval]]", "{}/{}".format(params.idx + 1, len(trainer.val_dataloader)), meter, fix=1)

    def on_save_checkpoint_end(self, trainer: Trainer, func, params: Params, result: str, *args, **kwargs):
        trainer.logger.info(f'Saved checkpoint in {result}')

    def on_save_model_end(self, trainer: Trainer, func, params: Params, result: str, *args, **kwargs):
        trainer.logger.info(f'Saved mdoel in {result}')

    def on_save_keypoint_end(self, trainer: Trainer, func, params: Params, result: str, *args, **kwargs):
        trainer.logger.info(f'Saved keypoint in {result}')

    def on_load_state_dict_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        res = self.getfuncargs(func, *args, **kwargs)
        object = res['object']
        if isinstance(object, str):
            trainer.logger.info(f'Load state dict from {object}')

        if meter is not None:
            trainer.logger.info(f'Loaded with meta info:')
            trainer.logger.raw(meter)
        else:
            trainer.logger.info(f'Loaded.')

    def on_imodels_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        trainer.logger.info('[[Model initialized]]')

    def on_prepare_dataloader_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        # print(meter)
        loader_id = id(meter)
        if loader_id in self.history_loader:
            return

        self.history_loader.add(loader_id)
        res = self.getfuncargs(func, *args, **kwargs)
        stage = res['stage'].name
        loader_ = trainer.datamodule[stage]
        if loader_ is None:
            trainer.logger.info(f'{stage.capitalize()} dataloader prepared but no dataloader created.')
        else:
            trainer.logger.info(f'{stage.capitalize()} dataloader prepared, size: {len(loader_)}.')


class MeterCheckpoint(TrainCallback):
    """
    Callback to save the model checkpoints by a monitored metric get it's best value.
    """
    only_main_process = True

    def __init__(self, monitor, mode="train", lower=True, start_epoch=0):
        self.monitor = monitor
        self.mode = mode
        self.lower = lower
        self.last_val = NoneItem()
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        self.update("train", trainer, params, meter)

    def update(self, cur_mode, trainer, param, meter):
        if cur_mode != self.mode:
            return
        if param.eidx > self.start_epoch:
            item = meter[self.monitor]
            if isinstance(item, AvgItem):
                item = item.avg
            if isinstance(item, NoneItem):
                return

            if self.lower:
                if self.last_val > item:
                    trainer.logger.info("model imporved from {} to {}".format(self.last_val, item))
                    trainer.save_checkpoint(meter.serialize())
                    self.last_val = item
            else:
                if self.last_val < item:
                    trainer.logger.info("model imporved from {} to {}".format(self.last_val, item))
                    trainer.save_checkpoint(meter.serialize())
                    self.last_val = item

    def on_test_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        self.update("test", trainer, params, meter)

    def on_eval_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        self.update("eval", trainer, params, meter)

    def __repr__(self) -> str:
        return self._repr_by_val("monitor", "mode", "lower", "start_epoch")


class EpochCheckpoint(TrainCallback):
    """
    在 Trainer 训练过程中定时保存模型
    """
    only_main_process = True

    def __init__(self, per_epoch=50):
        self.per_epoch = per_epoch

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        if params.eidx % self.per_epoch == 0 and params.eidx > 0:
            trainer.save_keypoint(meta_info=Meter.wrap_result(meter))

    def __repr__(self) -> str:
        return self._repr_by_val("per_epoch")


class GlobalStepCheckpoint(TrainCallback):
    only_main_process = True

    def __init__(self, per_step=2500):
        self.per = per_step

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, meter, *args, **kwargs)
        if params.global_step % self.per == 0 and params.global_step > 0:
            trainer.save_checkpoint(meta_info=Meter.wrap_result(meter))


class KeyErrorSave(TrainCallback):
    """
    Callback to save checkpoints when you interrupt the program.
    """
    only_main_process = True
    only_single_gpu = True
    priority = -1

    def __init__(self, wait_input=False):
        self.wait_input = wait_input

    def on_first_exception(self, source: Trainer, func, params: Params, e: BaseException, *args, **kwargs):
        if isinstance(e, (KeyboardInterrupt)):
            source.logger.info("KeyErrorSave trigged, save checkpoint")
            source.save_keypoint({"mode": "KeyboardInterrupt"})

            tp = "n"
            if self.wait_input:
                tp = input("continue train step? (y/other)")

            if tp.lower() == "y":
                return True


class ScalarRecorder(TrainCallback):
    """
    This callback logs scalars yield in train/test/eval by TensorBoard.
    """
    only_main_process = True
    priority = 100

    def __init__(self, writer=None) -> None:
        super().__init__()
        self.writer = writer

    def _key_name(self, mode, key):
        return "{}_{}_".format(key, mode)

    def write_scalar(self, meter, stage: str, global_step, writer):
        if isinstance(meter, (Meter, AvgMeter)):
            for k, v in meter.scalar_items():
                writer.add_scalar(self._key_name(stage, k), v, global_step)

    def on_test_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        writer = self.writer
        if writer is None:
            writer = trainer.writer
        self.write_scalar(meter, 'test', trainer.global_step, writer)

    def on_eval_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        writer = self.writer
        if writer is None:
            writer = trainer.writer
        self.write_scalar(meter, 'eval', trainer.global_step, writer)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        writer = self.writer
        if writer is None:
            writer = trainer.writer
        self.write_scalar(meter, 'train', trainer.global_step, writer)


class EMAUpdate(TrainCallback):
    """
    Callback to update EMA model every train step.

    Variable in Trainer instance is identified as EMA model when
     - is instance of torch.nn.Module
     - name is started with 'ema'
    """

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, meter, *args, **kwargs)
        for k, v in trainer.model_dict.items():
            if k.lower().startswith('ema'):
                v.step()


class AutoLoadModel(InitialCallback):
    """
    Callback to automatically load pretrained model.

    The load function will be executed if and only if
     - both the params `pretrain` and `pretrain_path` are defined in Params
     - `pretrain` is True and `pretrain_path` is not None.
    """

    def on_imodels_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_imodels_end(trainer, func, params, meter, *args, **kwargs)
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

    def on_imodels_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_imodels_end(trainer, func, params, meter, *args, **kwargs)
        if params.get('eval_first', True):
            if self.datamodule is not None:
                trainer.evaluate(self.datamodule)
            else:
                trainer.evaluate()


class RemoteMonitor(TrainCallback, InitialCallback, SaveLoadCallback):
    pass  # TODO


class StopByCode(TrainCallback):
    def __init__(self, step=100):
        self.step = step

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, meter, *args, **kwargs)
        if trainer.global_step % self.step == 0:
            if os.path.exists(trainer.exp.test_file('.stop')):
                trainer.exp.add_tag('lumo.early_stop')
                trainer.logger.info('Early stop this test manully.')
                trainer.stop_train_epoch()
                trainer.stop_train()
