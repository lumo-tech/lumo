from fire import Fire


class Interface:
    def init(self):
        pass

    def summary(self):
        pass

    def list(self):
        pass

    def latest(self):
        pass

    def info(self, test_name):
        pass

    def clear(self, cache=True):
        pass


if __name__ == '__main__':
    Fire(Interface)
