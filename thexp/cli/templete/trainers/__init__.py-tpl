from thexp import Params


class GlobalParams(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 400
        self.optim = self.create_optim('SGD',
                                       lr=0.06,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

        self.architecture = self.choice('architecture',
                                        'Lenet',
                                        'WRN', )
        self.dataset = self.choice('dataset', 'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn')
        self.n_classes = 10
        self.topk = (1, 2, 3, 4)

        self.batch_size = 64

        self.num_workers = 4

        self.ema = True
        self.ema_alpha = 0.999

        self.val_size = 10000

    def initial(self):
        if self.dataset in {'cifar100'}:
            self.n_classes = 100

        if self.ENV.IS_PYCHARM_DEBUG:
            self.num_workers = 0

        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00001,
                                     right=self.epoch)

    def wideresnet28_2(self):
        self.architecture = 'WRN'
        self.depth = 28
        self.widen_factor = 2

    def wideresnet28_10(self):
        self.architecture = 'WRN'
        self.depth = 28
        self.widen_factor = 10

