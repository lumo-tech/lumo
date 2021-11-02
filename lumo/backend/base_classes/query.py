"""
"""
import operator as op

import numpy as np


class Constrain():
    def __init__(self):
        self._name = None

    def __getattr__(self, item: str):
        if item.startswith('_'):
            return AttributeError(item)
        else:
            self._name = item
            return self

    def __getitem__(self, item: str):
        if item.startswith('_'):
            return AttributeError(item)
        self._name = item
        return self


class MeterConstrain(Constrain):
    """
    A meter constrain, mainly used for Boardquery.parallel()

    Examples:
    ```python
    from thexp import Q,C
    query = Q.tests().boards()
    print(query.scalar_tags)
    query.parallel(C.param.lr,C.param.epoch,C.meter.loss_train_,C.meter.acc_test_)
    ```
    """

    def __init__(self):
        super().__init__()
        self._constrain = None

    @property
    def max(self):
        self._constrain = np.max
        return self

    @property
    def min(self):
        self._constrain = np.min
        return self

    def __repr__(self):
        if self._constrain is not None:
            name = self._constrain.__name__.lstrip('a')
        else:
            name = '<empty>'

        return "Meter.{}({})".format(name, self._name)


_op_str = {
    'lt': '< ',
    'gt': '> ',
    'le': '<=',
    'ge': '>=',
    'eq': '==',
    'ne': '!=',
}


class ParamConstrain(Constrain):
    """
    A param constrain object corresponds to a constrain of a hyperparameter in Params class, which have a name,
    a comparison operator, a flag(whether can be none), and the comparison value.

    You can create this constrain by class ConstrainBuilder, which have a global instance named `C`

    >>> from thexp import C
    >>> constrain = (C.param.dataset == 'cifar10')
    ... print(constrain)

    Param(dataset == cifar10)


    >>> constrain = (C.param.allow_none.margin > 5)
    ... print(constrain)

    Param(Optional| margin >  5)


    You can use this instance in BoardQuery.parallel() to declare param name, or in TestQuery.filter_params() to query tests.

    Examples:
    ---------
    from thexp import Q
    Q.tests().filter_params(
        C.param.margin > 5,
        C.param.allow_none.margin > 5,
        C.param.epoch > 100,
        C.param.dataset == 'cifar10',
    )

    from thexp import Q
    Q.tests().boards().parallel(
        C.param.lr,
        C.meter.loss_train_,
        C.param.dataset
    )
    # for simpleness, you can create name with C.param_names('lr','margin',...)
    Q.tests().boards().parallel(
        *C.param_names('lr','margin'),
    )

    """

    def __init__(self):
        super().__init__()
        self._constrain = None
        self._allow_none = False
        self._value = None

    def __lt__(self, other):
        self._constrain = op.lt
        self._value = other
        return self

    def lt(self, other):
        self._constrain = op.lt
        self._value = other
        return self

    def __gt__(self, other):
        self._constrain = op.gt
        self._value = other
        return self

    def gt(self, other):
        self._constrain = op.gt
        self._value = other
        return self

    def __le__(self, other):
        self._constrain = op.le
        self._value = other
        return self

    def le(self, other):
        self._constrain = op.le
        self._value = other
        return self

    def __ge__(self, other):
        self._constrain = op.ge
        self._value = other
        return self

    def ge(self, other):
        self._constrain = op.ge
        self._value = other
        return self

    def __eq__(self, other):
        self._constrain = op.eq
        self._value = other
        return self

    def eq(self, other):
        self._constrain = op.eq
        self._value = other
        return self

    def __ne__(self, other):
        self._constrain = op.ne
        self._value = other
        return self

    def ne(self, other):
        self._constrain = op.ne
        self._value = other
        return self

    def __repr__(self):
        if self._allow_none:
            off = 'Optional| '
        else:
            off = ''

        return 'Param({}{} {} {})'.format(off, self._name, _op_str[self._constrain.__name__], self._value)

    @property
    def allow_none(self):
        self._allow_none = True
        return self


class ConstrainBuilder():
    @property
    def param(self):
        return ParamConstrain()

    @property
    def meter(self):
        return MeterConstrain()

    def param_names(self, *names: str):
        return [ParamConstrain()[name] for name in names]

    def meter_names(self, *names: str):
        return [MeterConstrain()[name] for name in names]


C = ConstrainBuilder()
