from black import main
from lumo import Params


class TrainParams(Params):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.epoch = 100
        
        
class DatasetParams(Params):
    def __init__(self):
        super().__init__()
        self.dataset = self.choice('cifar10','cifar100')
    
class OptimParams(Params):
    def __init__(self):
        super().__init__()
        self.name = 'SGD'
        self.lr = 0.01
        self.moment = 0.9
        
class MyParams(TrainParams,DatasetParams):
    def __init__(self):
        super().__init__()
        self.optim = OptimParams()
    

# python extends.py --optim.lr=0.2
if __name__ == '__main__':
    params = MyParams()
    params.from_args()
    print(params)