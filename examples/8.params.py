"""

"""

from lumo import Params
params = Params()
params.from_args()
print(params)

params.epoch = 400
params.batch_size = 25
print(params)


from lumo import Params
class MyParams(Params):
    def __init__(self):
        super().__init__()
        self.batch_size = 50
        self.topk = (1,2,3,4)
        self.optim = dict(
            lr=0.009,
            moment=0.9
        )

params = MyParams()
print(params)


from lumo import Params
params = Params()
params.choice("dataset","mnist","cifar10","cifar100","svhn")
params.arange("thresh",5,0,20)
print(params)

# for g in params.grid_search("thresh",range(0,20)):
#     for g in g.grid_search("dataset",['cifar10','cifar100','svhn']):
#         print(g.dataset,g.thresh)


params.bind('dataset','mnist','arch','simplenet')
params.bind('dataset','cifar10','arch','cnn13')
params.bind('arch','simplenet','arch_param',dict(feature=128))
params.bind('arch','cnn13','arch_param',dict(feature=256))
params.dataset = 'cifar10'
print(params.arch)
print(params.arch_param)
params.dataset = 'mnist'
print(params.arch)
print(params.arch_param)



params.to_json('params.json')
params.from_json('params.json')


p = Params()
p.margin = 0.5
p.margin = p.default(0.1,True)


p = Params()
p.margin = p.default(0.3,True)
print(p.margin)

p.optim = p.create_optim('SGD',lr=0.1)
from torch import nn
p.optim.build(nn.Linear(2,2).parameters())
cp = p._param_dict.copy()



print(cp)

