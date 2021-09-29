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
from lumo.contrib.optim.lr_scheduler import *
