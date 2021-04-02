class Value():
    def __init__(self, value, name=None):
        self.value = value
        self.name = name

    def __call__(self, value, name=None):
        self.value = value
        self.name = name


class Loss(Value):
    pass


class Metric(Value):
    pass
