from torch import nn

from thexp import Trainer
from thexp.contrib import EMA, ParamGrouper

import arch
from .. import GlobalParams
import torch


def load_backbone(params: GlobalParams):
    if params.architecture == 'WRN':
        params.with_fc = params.default(True)
        params.depth = params.default(28)
        params.widen_factor = params.default(2)
        params.drop_rate = params.default(0)

        from arch.wideresnet import WideResNet
        model = WideResNet(depth=params.depth,
                           widen_factor=params.widen_factor,
                           with_fc=params.with_fc,
                           dropout_rate=params.drop_rate,
                           num_classes=params.n_classes)

    elif params.architecture == 'Resnet':
        from torchvision.models import resnet
        raise NotImplementedError()

    elif params.architecture == 'Lenet':
        from arch import lenet
        params.with_fc = params.default(True)
        model = lenet.LeNet(params.n_classes, with_fc=params.with_fc)
    else:
        assert False

    return model


class ModelMixin(Trainer):

    def predict(self, xs):
        raise NotImplementedError()

    def models(self, params: GlobalParams):
        raise NotImplementedError()


class BaseModelMixin(ModelMixin):
    """base end-to-end model"""

    def predict(self, xs) -> torch.Tensor:
        with torch.no_grad():
            if self.params.ema:
                model = self.ema_model
            else:
                model = self.model
            if not self.params.with_fc:
                return model.fc(model(xs))
            return model(xs)

    def models(self, params: GlobalParams):
        model = load_backbone(params)

        if params.distributed:
            from torch.nn.modules import SyncBatchNorm
            model = SyncBatchNorm.convert_sync_batchnorm(model.cuda())
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank])
        else:
            self.model = model

        if params.ema:
            self.ema_model = EMA(self.model)

        grouper = ParamGrouper(self.model)
        noptim = params.optim.args.copy()
        noptim['weight_decay'] = 0
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]

        self.optim = params.optim.build(param_groups)
        if not params.distributed:
            self.to(self.device)
