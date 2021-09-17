"""

"""

import os
import sys
from functools import wraps
import inspect

from .mixin import CallbackMix
from .meter import AvgMeter
from .meter import Meter
from .params import Params
from ..base_classes.trickitems import NoneItem, AvgItem
from ..calculate.schedule import Scheduler, SchedulerList
from ..utils.timing import format_second
from typing import TYPE_CHECKING

func_map = {
    'evaluate': 'eval',
    'evaluate_step': 'eval_step'
}

from .trainer import Trainer, TrainerResult


# if TYPE_CHECKING:


def map_func_name(name):
    return func_map.get(name, name)


class BaseCallback():
    """
    base callback class

    only have two methods `on_begin()` and `on_end()`.

    for simpler using, see TrainCallback.
    """
    priority = 0  # type:int # All callbacks in lumo will have priority in range 0-100
    only_single_gpu = False  # only hooked in single gpu mode
    only_main_process = False  # whether can be hooked in children process( local_rank > 0)

    def __new__(cls, *_, **__):
        self = super().__new__(cls)
        self._hooked = None

        def ecp_wrap(func):
            """同一个异常第一次调用的时候运行"""

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
    实现了一般训练过程中的函数函数回调，主要将 on_begin() / on_end() 方法分发到各具体的回调方法中
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
    """
    only_main_process = True

    def __init__(self, eval_per_epoch=1, test_per_epoch=10):
        self.eval_in_per_epoch = eval_per_epoch
        self.test_in_per_epoch = test_per_epoch

        # evaluate/test on train end
        self._last_eval = -1
        self._last_test = -1

    def _test_or_eval(self, params: Params, trainer: Trainer):
        if self.eval_in_per_epoch is not None and self.eval_in_per_epoch > 0:
            if params.eidx % self.eval_in_per_epoch == self.eval_in_per_epoch - 1:
                self._last_eval = params.eidx
                trainer.evaluate()
        if self.test_in_per_epoch is not None and self.test_in_per_epoch > 0:
            if params.eidx % self.test_in_per_epoch == self.test_in_per_epoch - 1:
                self._last_test = params.eidx
                trainer.test()

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        self._test_or_eval(params, trainer)

    def on_train_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        if self._last_eval != params.eidx:
            trainer.evaluate()
        if self._last_test != params.eidx:
            trainer.test()

    def __repr__(self):
        return self._repr_by_val("eval_in_per_epoch", "test_in_per_epoch")


class LoggerCallback(TrainCallback, InitialCallback, SaveLoadCallback):
    """
    用于日志输出的回调，当 BaseTrainer 在 epoch / batch 等级别的训练结束、异常发生等过程后，Logger 会对这些事件，
    或方法返回的结果进行输出。

    一般情况下 Logger 支持所有类型输出，但如果使用 Meter 类进行包装，会有更好的输出形式
    """
    only_main_process = True
    priority = 100

    def __init__(self, avg=True, step_frequence=3, breakline_in=1000):
        self.avg = avg
        self.step_frequence = step_frequence
        self.breakline = breakline_in

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
        return self.step_frequence > 0 and step % self.step_frequence == 0

    def _need_breakline(self, step):
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
        from ..utils.timing import TimeIt

        self.start = params.eidx
        self.traintime = TimeIt()
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
        from ..utils.timing import TimeIt
        self.reset_meter()
        self.epochtime = TimeIt()
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
        trainer.logger.info(tm)

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        meter = trainer._wrap_result(meter)
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
        self.reset_meter()
        trainer.logger.info("[[Test Begin]]")

    def on_test_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        if meter is None:
            meter = ""
        trainer.logger.info("[[Test End]]", meter)

    def on_test_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        meter = trainer._wrap_result(meter)
        self.meter.update(meter)
        meter = self.meter
        trainer.logger.inline("{}/{}".format(params.idx + 1, len(trainer.test_dataloader)), meter, fix=1)

    def on_eval_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.reset_meter()
        trainer.logger.info("[[Eval Begin]]")

    def on_eval_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        meter = result.meter
        if meter is None:
            meter = ""
        trainer.logger.info("[[Eval End]]", meter)

    def on_eval_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        meter = trainer._wrap_result(meter)
        self.meter.update(meter)
        meter = self.meter
        trainer.logger.inline("{}/{}".format(params.idx + 1, len(trainer.val_dataloader)), meter, fix=1)

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
        res = self.getfuncargs(func, *args, **kwargs)
        stage = res['stage'].name
        loader_ = trainer.datamodule[stage]
        if loader_ is None:
            trainer.logger.info(f'{stage.capitalize()} dataloader prepared but no dataloader created.')
        else:
            trainer.logger.info(f'{stage.capitalize()} dataloader prepared, size: {len(loader_)}.')


class MeterCheckpoint(TrainCallback):
    """
    用于检视训练过程中模型的某个指标，并根据其提升进行 checkpoint 类型的保存
    该类参考了 Keras 中相应的实现。
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
            trainer.save_keypoint(meta_info=trainer._wrap_result(meter))

    def __repr__(self) -> str:
        return self._repr_by_val("per_epoch")


class GlobalStepCheckpoint(TrainCallback):
    only_main_process = True

    def __init__(self, per_step=2500):
        self.per = per_step

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, meter, *args, **kwargs)
        if params.global_step % self.per == 0 and params.global_step > 0:
            trainer.save_checkpoint(meta_info=trainer._wrap_result(meter))


class KeyErrorSave(TrainCallback):
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


# class BoardRecord(TrainCallback):
#     """
#     自动记录训练过程中的所有变量到 tensorboard 中（epoch 级）
#     """
#     only_main_process = True
#     priority = 100
#
#     def __init__(self) -> None:
#         super().__init__()
#
#     def on_hooked(self, source: Trainer, params: Params):
#         self.start = 0
#
#     def _key_name(self, mode, key):
#         return "{}_{}_".format(key, mode)
#
#     def on_test_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
#         meter = result.meter
#         if isinstance(meter, Meter):
#             for k, v in meter.numeral_items():
#                 trainer.writer.add_scalar(self._key_name("test", k), v, params.eidx)
#
#     def on_train_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
#         self.start = params.eidx
#
#     def on_eval_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
#         meter = result.meter
#         if isinstance(meter, Meter):
#             for k, v in meter.numeral_items():
#                 trainer.writer.add_scalar(self._key_name("eval", k), v, params.eidx)
#
#     def on_train_epoch_end(self, trainer: Trainer, func, params: Params, meter: AvgMeter, *args, **kwargs):
#         if isinstance(meter, Meter):
#             for k, v in meter.numeral_items():
#                 trainer.writer.add_scalar(self._key_name("train", k), v, params.eidx)


class EMAUpdate(TrainCallback):
    only_main_process = True

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, meter, *args, **kwargs)
        for k, v in trainer.model_dict.items():
            if k.lower().startswith('ema'):
                v.step()


# class LRSchedule(TrainCallback):
#     def __init__(self, schedule: Schedule = None, apply=True, use_eidx=True):
#         self.schedule = schedule
#         self.apply = apply
#         self.use_eidx = use_eidx
#
#     def on_hooked(self, source: Trainer, params: Params):
#         super().on_hooked(source, params)
#         if self.schedule is None:
#             if 'lr_sche' not in params:
#                 source.logger.warn(f'lr_sche not exists in params and be assigned, {self} will be unhooked after.')
#                 self.unhook()
#             else:
#                 self.schedule = params.lr_sche
#
#     def on_train_epoch_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
#         for k, v in trainer.optim_dict.items():
#             if self.use_eidx:
#                 step = params.eidx
#             else:
#                 step = params.global_step
#
#             if self.apply:
#                 new_lr = self.schedule.apply(v, step)
#                 trainer.logger.info('{}.lr = {}'.format(k, new_lr))
#             else:
#                 ratio = self.schedule.scale(v, step)
#                 trainer.logger.info('lr scale ratio = {}'.format(k, ratio))


# class ReportSche(TrainCallback):
#     """
#     log `schedule` in every epoch end
#     `schedule` means `Schedule` in Params and have `sche` in the name, which will have different value in every epoch
#     """
#     only_main_process = True
#     priority = 100
#
#     def on_hooked(self, source: Trainer, params: Params):
#         self.sche_lis = []
#         for k, v in params.items():  # type:str, Any
#             if isinstance(v, (Schedule, ScheduleList)) and 'sche' in k.lower():
#                 self.sche_lis.append((k, v))
#
#     def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
#         super().on_train_epoch_end(trainer, func, params, result, *args, **kwargs)
#         meter = result.meter
#         m = Meter()
#         for k, v in self.sche_lis:
#             m[k] = v(params.eidx)
#         trainer.logger.info(m)


class AutoLoadModel(InitialCallback):

    def on_imodels_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_imodels_end(trainer, func, params, meter, *args, **kwargs)
        if params.get('pretrain', False):
            path = params.get('pretrain_path', None)
            if path is not None:
                trainer.load_state_dict(path)


class EvalFirst(AutoLoadModel):

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
