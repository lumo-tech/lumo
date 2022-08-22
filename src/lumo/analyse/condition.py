from operator import and_, gt, ge, le, lt, eq, ne
from itertools import accumulate


def in_(ser, value):
    return ser.apply(lambda x: x in value)


def not_in_(ser, value):
    return ser.apply(lambda x: x not in value)


def first(ser, value):
    return ser.duplicated(value) == False


mapping = {
    '>=': ge,
    '<=': le,
    '==': eq,
    '!=': ne,
    '>': gt,
    '<': lt,
    'in': in_,
    'notin': not_in_,
    'first': first,
}


class Compare:
    def __init__(self, name=None):
        self.name = name
        self.op = None
        self.value = None
        self.drop = False

    def __getattr__(self, item):
        return Compare(item)

    def __getitem__(self, item):
        return Compare(item)

    def __invert__(self):
        self.drop = ~self.drop
        return self

    def __neg__(self):
        self.drop = True
        return self

    def __ge__(self, other):
        assert other is not None
        self.value = other
        self.op = ">="
        return self

    def __le__(self, other):
        assert other is not None
        self.value = other
        self.op = "<="
        return self

    def __eq__(self, other):
        self.value = other
        self.op = "=="
        return self

    def __ne__(self, other):
        self.value = other
        self.op = "!="
        return self

    def __gt__(self, other):
        assert other is not None
        self.value = other
        self.op = ">"
        return self

    def __lt__(self, other):
        assert other is not None
        self.value = other
        self.op = "<"
        return self

    def __repr__(self):
        return f'{self.name} {self.op} {self.value}'

    def in_(self, lis):
        self.op = 'in'
        self.value = set(lis)
        return self

    def not_in_(self, lis):
        self.op = 'notin'
        self.value = set(lis)
        return self

    def first(self, value):
        self.op = 'first'
        self.value = value
        return self


C = Compare()


def filter_by_condition(df, *condition: Compare):
    filters = []
    remains = []
    drops = []
    for cond in condition:
        if cond.op is None:
            if cond.drop:
                drops.append(cond.name)
            else:
                remains.append(cond.name)
        else:
            if cond.value is None:
                if cond.op == '==':
                    res = df[cond.name].isna()
                else:
                    res = ~df[cond.name].isna()
            else:
                op = mapping[cond.op]
                res = op(df[cond.name], cond.value)
            filters.append(res)

    df = df[list(accumulate(filters, and_))[-1]]

    if len(remains) > 0:
        df = df[remains]
    if len(drops) > 0:
        df = df.drop(drops, axis=1)

    return df
