from thexp import Params
from thexp.frame import BaseParams
import os
import pickle


def create_params():
    params = BaseParams()
    params.optim = params.create_optim('SGD', lr=0.1, momentum=0.9)
    params.sche = params.SCHE.Cos()
    params.schelis = params.SCHE.List(
        [params.SCHE.Cos(left=0, right=6),
         params.SCHE.Linear(left=6, right=10, start=1), ]
    )
    params.dataset = params.choice('dataset', 'cifar10', 'cifar100', 'svhn', 'stl10')
    params.n_classes = 10
    params.margin = params.arange('margin', 1.4, 0.5, 2)

    params.dataset = 'cifar100'
    params.new2 = 5
    return params


def test_baseparams():
    params = create_params()

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

    params.new1 = params.default(5)
    assert params.new1 == 5

    params.new2 = params.default(6)
    assert params.new2 == 5

    assert 'new3' not in params
    with open('a.pkl', 'wb') as w:
        pickle.dump(params._param_dict, w)
    with open('a.pkl', 'rb') as r:
        _param_dict = pickle.load(r)
    assert params._param_dict == _param_dict

    with open('b.pkl', 'wb') as w:
        pickle.dump(params, w)
    with open('b.pkl', 'rb') as r:
        params3 = pickle.load(r)
    assert params == params3

    os.remove('a.pkl')
    os.remove('b.pkl')


def test_deafult_warn():
    import warnings

    params = create_params()
    with warnings.catch_warnings(record=True) as w:
        params.k = params.default(10, True)
        params.k2 = params.default(11, True)
        assert len(w) == 2




def test_params():
    pass
