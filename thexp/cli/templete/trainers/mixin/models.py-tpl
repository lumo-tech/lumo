from torch import nn

from thexp import Trainer
from thexp.contrib import EMA, ParamGrouper

import arch
from .. import GlobalParams


class ModelMixin(Trainer):

    def models(self, params: GlobalParams):
        raise NotImplementedError()

    def predict(self, xs):
        raise NotImplementedError()


class TempleteModelMixin(ModelMixin):
    def models(self, params: GlobalParams):
        self.model = nn.Linear(2, params.n_classes)

        if params.ema:
            self.ema_model = EMA(self.model, alpha=params.ema_alpha)

        grouper = ParamGrouper(self.model)

        noptim_args = params.optim.args.copy()
        noptim_args['weight_decay'] = 0
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim_args),
            grouper.create_param_group(grouper.norm_params(), **noptim_args)
        ]
        self.optim = params.optim.build(param_groups)

        self.to(self.device)

    def predict(self, xs):
        import torch
        return self.model(torch.rand(xs.shape[0], 2))
