"""
This module aims to provide an easy and general method to generate series values, which can be used in
 applying learning rate of optimizer or scaling a loss value as weight.

To meet this demand, the concept 'schedule' is summarized as a periodic math function, which have
left/right interval and  start/end value.

Hence, a `Schedule` class is provided, it receives four base parameters: start, end, left, and right corresponding to
the above concept respectively.

This class provides some common methods. For example, when you have a schedule instance, you can apply learning rate by
simple call `schedule.scale()` or `schedule.apply()` function.

And, you can use `schedule.plot()` to plot a curve of the values in each step. The plot function use `matplotlib` as backend.

If you don't want to create an instance, you can call classmethod `get(cls, cur, start=0, end=1, left=0, right=1)` to get
value.

Except for the base `Schedule` class, some other subclasses of `Schedule`  which may be general used is provided, too. All can
 be easily understand by their names and plot curves.
"""
from typing import List

import numpy as np

from lumo.core import BaseParams
from lumo.core.metaclasses import PropVar

__all__ = ['Cos',
           'Linear',
           'Exp',
           'Log',
           'Constant',
           'PeriodCos',
           'PeriodHalfCos',
           'PeriodTriangle',
           'PeriodLinear',
           'PowerDecay',
           'InterpolateList', ]


class Interpolate(BaseParams):
    """
    ratio 变化为从 1 - 0
    """

    def toggle_constant(self, toggle=True):
        """fix the schedule as the first value"""
        self.constant = toggle
        return self

    @classmethod
    def interp(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        content = ', '.join(["{}={}".format(k, v) for k, v in self.items()])
        return "{}({})".format(self.__class__.__name__, content)

    def __call__(self, cur):
        raise NotImplementedError()

    def plot(self, num=1000, left=0, right=1000, show=True):
        """
        Plot a curve of the schedule.
        Args:
            num:
            left:
            right:

        Returns:
            plt.plot

        Notes:
            You may need to call `plt.show`() to show it.
        """
        from matplotlib import pyplot as plt

        x = np.linspace(left, right, num)
        y = [self(i) for i in x]

        res = plt.plot(x, y)
        plt.title(str(self))
        if show:
            plt.show()
        return res

    def scale(self, optimizer, cur):
        """
        Scale the learning rate by current value. 'scale' means not apply the current schedule value directly to
        the learning rate, but multiple the initial learning rate. You can use `schedule.apply()` to apply the schedule
        value directly.

        Notes:
        -------
        When first apply scale function, a `_raw_lr` represent initial lr will be set in each param_group, then, the
        learning rate(store in param_groups with key 'lr') will be calculated by `_raw_lr * schedule(cur)`.

        Args:
            optimizer: A pytorch optimizer instance.
            cur: current step of this schedule.

        Returns:
            Current schedule value.
        """
        ratio = self(cur)
        for param_group in optimizer.param_groups:  # type:dict
            raw_lr = param_group.setdefault('_raw_lr', param_group['lr'])
            param_group['lr'] = raw_lr * ratio

        return ratio

    def apply(self, optimizer, cur):
        """
        Apply the learning rate with current schedule value.

        Args:
            optimizer: A pytorch optimizer instance.
            cur: current step of this schedule.

        Returns:

        """
        new_lr = self(cur)
        for param_group in optimizer.param_groups:  # type:dict
            param_group['lr'] = new_lr

        return new_lr


class ABCContinuous(Interpolate):

    def __call__(self, cur):
        if self.constant:
            return self.start
        return self.interp(cur, start=self.start, end=self.end, left=self.left, right=self.right,
                           constant=self.constant)

    def __init__(self, start=1e-3, end=1e-6, left=0, right=80, *args, **kwargs):
        super().__init__()
        self.left = left
        self.right = right
        self.start = start
        self.end = end
        self.constant = kwargs.get('constant', (left == right))

    @classmethod
    def ratio(cls, cur, left, right, constant=False):
        if constant:
            return 0
        return (cur - left) / (right - left)

    @classmethod
    def get_val(cls, cur, start=1e-3, end=1e-6, left=0, right=80, *args, **kwargs):
        """get the current schedule value without create `schedule` instance. """
        return cls.interp(cur=cur, start=start, end=end, left=left, right=right, *args, **kwargs)

    def plot(self, num=1000, left=None, right=None, show=True):
        if left is None:
            left = self.left

        if right is None:
            right = self.right

        if self.constant:
            right = left + 50

        return super().plot(num, left, right, show)


class ABCPeriod(Interpolate):
    """
    period
    """

    def __init__(self, start=0, end=1, left=0, period=1, *args, **kwargs):
        super().__init__()
        self.left = left
        self.period = period
        self.start = start
        self.end = end
        self.constant = (period == 0)

    @classmethod
    def ratio(cls, cur, left, period, constant=False):
        if constant:
            return 0
        if cur < left:
            in_period = (period - (left - cur) % period) % period
        else:
            in_period = float(cur - left) % period
        return in_period / period

    @classmethod
    def get_val(cls, cur, start=0, end=1, left=0, right=1, *args, **kwargs):
        """get the current schedule value without create `schedule` instance. """
        return cls.interp(cur=cur, start=start, end=end,
                          left=left, right=right, *args, **kwargs)

    def plot(self, num=1000, left=None, n_period=5, show=True):
        if left is None:
            left = self.left

        right = self.period * n_period
        if self.constant:
            right = left + 50

        return super().plot(num, left, right, show)

    def __call__(self, cur):
        return self.interp(cur, start=self.start, end=self.end, left=self.left, period=self.period,
                           constant=self.constant)


class Cos(ABCContinuous):
    """one cycle cosine functoin"""

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., right=1., *args, **kwargs):
        constant = kwargs.get('constant', False)
        if constant:
            return start

        if cur < left:
            return start
        elif cur > right:
            return end

        ratio = cls.ratio(cur, left=left, right=right, constant=constant)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi))
        return start * cos_ratio + end * (1 - cos_ratio)

    def get(self, cur):
        return self(cur)


class Linear(ABCContinuous):
    """linear schedule"""

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., right=1., *args, **kwargs):
        constant = kwargs.get('constant', False)
        if constant:
            return start

        if cur < left:
            return start
        elif cur > right:
            return end

        linear_ratio = cls.ratio(cur, left=left, right=right, constant=constant)
        return start * (1 - linear_ratio) + end * linear_ratio


class Exp(ABCContinuous):
    """slow to quick"""

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., right=1., *args, **kwargs):
        constant = kwargs.get('constant', False)
        if constant:
            return start

        if cur < left:
            return start
        elif cur > right:
            return end

        ratio = cls.ratio(cur, left=left, right=right, constant=constant)
        residual = np.exp(-5)

        exp_ratio = np.exp((ratio - 1) * 5) - residual * (1 - ratio)
        return start * (1 - exp_ratio) + end * exp_ratio


class Log(ABCContinuous):
    """quick to slow"""

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., right=1., *args, **kwargs):
        constant = kwargs.get('constant', False)
        if constant:
            return start

        if cur < left:
            return start
        elif cur > right:
            return end

        ratio = cls.ratio(cur, left=left, right=right, constant=constant)

        residual = np.exp(-5)

        log_ratio = 1 - np.exp(-ratio * 5) + residual * ratio
        return start * (1 - log_ratio) + end * log_ratio


class Constant(ABCContinuous):
    def __init__(self, value=0.5, *args, **kwargs):
        super().__init__(start=value, end=value, left=0, right=1, *args, **kwargs)
        self.constant = True


class PeriodCos(ABCPeriod):
    """
    periodic cosine schedule
    """

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., period=1., *args, **kwargs):
        constant = kwargs.get('constant', False)

        ratio = cls.ratio(cur, left=left, period=period, constant=constant)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi * 2))
        return start * cos_ratio + end * (1 - cos_ratio)


class PeriodHalfCos(ABCPeriod):
    """
    half periodic cosine schedule, period is (right-left)
    """

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., period=1., *args, **kwargs):
        constant = kwargs.get('constant', False)
        ratio = cls.ratio(cur, left=left, period=period, constant=constant)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi))
        return start * cos_ratio + end * (1 - cos_ratio)


class PeriodTriangle(ABCPeriod):
    def __init__(self, start=0, end=1, left=0, left_period=1, right_period=1, *args, **kwargs):
        super().__init__(start=start, end=end, left=left,
                         period=(left_period + right_period),
                         *args, **kwargs)
        assert left_period > 0 and right_period > 0
        self.left_period = left_period
        self.right_period = right_period

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., left_period=0., right_period=1., *args, **kwargs):
        constant = kwargs.get('constant', False)
        ratio = cls.ratio(cur, left=left, period=(left_period + right_period), constant=constant)

        mid_ratio = left_period / (right_period + left_period)

        if ratio < mid_ratio:
            ratio = ratio / mid_ratio
            return start * (1 - ratio) + end * ratio
        else:
            ratio = (ratio - mid_ratio) / (1 - mid_ratio)
            return end * (1 - ratio) + start * ratio


class PeriodLinear(ABCPeriod):
    """
    sawtooth wave, like a period line schedule
    """

    @classmethod
    def interp(cls, cur, start=0., end=1., left=0., period=1., *args, **kwargs):
        constant = kwargs.get('constant', False)
        ratio = cls.ratio(cur, left=left, period=period, constant=constant)
        return start * (1 - ratio) + end * ratio


class PowerDecay(Interpolate):
    """equal to tf.train.exponential_decay, decay every <decay_steps> with a base of <decay_rate> """

    def __init__(self, start, decay_steps, decay_rate, end=None):
        super().__init__()
        self.start = start
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.end = end

    def __call__(self, cur):
        rate = np.power(self.decay_rate, cur // self.decay_steps)
        res = self.start * rate
        if self.end is not None:
            res = max(self.end, res)

        return res


class PowerDecay2(Interpolate):
    def __init__(self, start, schedules, gammas):
        super().__init__()
        self.start = start
        self.schedules = schedules
        self.gammas = gammas

    @classmethod
    def interp(cls, cur, start=0., gammas=None, schedules=None, *args, **kwargs):
        if schedules is None:
            schedules = []
        if gammas is None:
            gammas = []

        res = start
        for (gamma, step) in zip(gammas, schedules):
            if (cur >= step):
                res = res * gamma
            else:
                break
        return res

    def __call__(self, cur):
        self.interp(cur, start=self.start, gammas=self.gammas, schedules=self.schedules)


class InterpolateList(Interpolate):
    def __init__(self, schedules: List[Interpolate]):
        super().__init__()
        self.schedules = schedules
        if len(schedules) > 1:
            for i in range(len(schedules) - 1):
                if schedules[i].right < schedules[i + 1].left:
                    raise ValueError('schedule[i].right should equal or larger than schedule[i+1].left, '
                                     f'but got {schedules[i].right} and {schedules[i + 1].left}')

            self.left = self.schedules[0].left
            self.right = self.schedules[-1].right

    def __reduce__(self):
        return (self.__class__, (self.schedules, self.bound))

    def __call__(self, cur):
        for i, schedule in enumerate(self.schedules):
            if i + 1 < len(self.schedules):
                if cur < self.schedules[i + 1].left:
                    return schedule(cur)
                else:
                    continue
            else:
                return schedule(cur)

    def __repr__(self):
        content = ', '.join([i.__class__.__name__ for i in self.schedules])
        return '{}({})'.format(self.__class__.__name__, content)

    def plot(self, num=1000, left=None, right=None, show=True):
        if left is None:
            left = self.left

        if right is None:
            right = self.right

        return super().plot(num, left, right, show)
