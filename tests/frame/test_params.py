from thexp import Params
from thexp.frame import BaseParams
import os


def test_baseparams():
    params = BaseParams()
    params.optim = params.create_optim('SGD', lr=0.1, momentum=0.9)
    params.sche = params.SCHE.Cos()
    params.schelis = params.SCHE.List(
        [params.SCHE.Cos(left=0, right=6),
         params.SCHE.Linear(left=6, right=10, start=1), ]
    )
    params.dataset = params.choice('dataset', 'cifar10', 'cifar100', 'svhn', 'stl10')
    params.n_classes = 10
    params.bind('dataset', 'cifar100', 'n_classes', 100)
    params.margin = params.arange('margin', 1.4, 0.5, 2)

    params.dataset = 'cifar100'
    assert params.n_classes == 100

    params.to_json('tmp')
    params2 = BaseParams()
    params2.from_json('tmp')
    os.remove('tmp')

    assert params == params2

    assert params._repeat == None
    for i, subp in enumerate(params.grid_range(5)):
        assert subp._repeat == i
    assert params._repeat == None

    iter_dataset = ['cifar10', 'cifar100']
    iter_lr = [0.3, 0.1, 0.03, 0.01]
    for i, subp in enumerate(params.grid_search('dataset', iter_dataset)):
        for j, subpp in enumerate(subp.grid_search('optim.args.lr', iter_lr)):
            assert subpp.dataset == iter_dataset[i]
            assert subpp.optim.args.lr == iter_lr[j]

    assert params == params2

    assert params.optim.args.lr == 0.1
    params.dynamic_bind('batch_size', 'optim.args.lr', lambda x: 3 / x)
    params.batch_size = 128
    assert params.optim.args.lr == 3 / params.batch_size


def test_params():
    pass
