from lumo import Params
from lumo.base_classes import attr
from lumo.base_classes.params_vars import OptimBuilder


class A(Params):

    def __init__(self):
        super().__init__()
        self.temp = None
        self.op = self.OPTIM.create_optim('Adam', lr=1)
        self.k = self.arange(2)  # asd


print(A())
aa = A.Space(
    epoch=10,  # int
    eidx=0,  # int
    idx=0,  # int
    global_step=0,  # int
    device='cpu',  # str
    stage='init',  # choice_param(default='init', choices=('init', 'train', 'test', 'val')), str
    temp=None,  # NoneType
    op=OptimBuilder([('name', 'Adam'), ('lr', 1)]),  # OptimBuilder
    k=2,  # arange_param(default=2, left=-inf, right=inf), int
)
aa.from_args()
print(aa)
