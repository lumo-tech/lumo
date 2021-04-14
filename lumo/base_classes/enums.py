from enum import Enum


class TrainerStage(Enum):
    init = -1
    train = 0
    test = 1
    evaluate = 2

    @property
    def is_train(self):
        return self == self.train

    @property
    def is_test(self):
        return self == self.test

    @property
    def is_eval(self):
        return self == self.evaluate


