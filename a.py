from lumo import Params


class A(Params):

    def __init__(self):
        super().__init__()
        self.temp = None
        self.op = self.OPTIM.create_optim('Adam', lr=1)
        self.k = self.arange(2)  # asd



print(A())
