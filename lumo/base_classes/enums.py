from enum import Enum


class TrainerStage(Enum):
    debug = -2
    init = -1
    train = 0
    train_epoch = 1
    test = 2
    val = 3

    @property
    def is_train(self):
        return self == self.train

    @property
    def is_test(self):
        return self == self.test

    @property
    def is_eval(self):
        return self == self.val
