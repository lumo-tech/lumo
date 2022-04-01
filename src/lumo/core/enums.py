import enum


class TrainStage(enum.Enum):
    default = 'default'
    train = 'train'
    test = 'test'
    val = 'val'

    def is_train(self):
        return self.value == 'train'

    def is_test(self):
        return self.value == 'test'

    def is_val(self):
        return self.value == 'val'

    @staticmethod
    def create_from_str(value):
        if value in {'eval', 'evaluate'}:
            value = 'val'
        return TrainStage(value)
