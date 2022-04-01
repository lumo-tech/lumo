from typing import Tuple, Union, List

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


def walk_module(module: Union[Tuple[nn.Module, ...], nn.Module]) -> Tuple[nn.Module, str, nn.Parameter]:
    if isinstance(module, (tuple, list)):
        for item in module:
            for ssubmodule, subname, subparam in walk_module(item):
                yield ssubmodule, subname, subparam
    else:
        for name, submodule in module.named_children():
            for ssubmodule, subname, subparam in walk_module(submodule):
                yield ssubmodule, subname, subparam

        for pname, param in module.named_parameters(recurse=False):
            yield module, pname, param


class ParamGrouper:
    """
    Most of pytorch Optimizers have the hyperparam `weight decay`, but if you use `model.parameters()` as the
     Optimizer's parameter, it will decay all parameters including bias,

    《Bag of Tricks for Image Classification with Convolutional Neural Networks》 suggests that only
    decay weight of linear or convolution layer.

    The best practice to do this in pytorch is to build param_groups for the parameter of Optimizer, and this class
    can do this easily and elegant.

    Examples:
    >>> grouper = ParamGrouper(model)
    ... noptim = params.optim.args.copy()
    ... noptim['weight_decay'] = 0
    ... param_groups = [
    ...     grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
    ...     grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
    ...     grouper.create_param_group(grouper.norm_params(), **noptim),
    ... ]
    """

    def __init__(self, *module: nn.Module):
        self.module = module

    def params(self, with_norm=True):
        params = []
        for module, name, param in walk_module(self.module):
            if with_norm or not isinstance(module, (_BatchNorm, nn.LayerNorm)):
                params.append(param)
        return params

    def kernel_params(self, with_norm=False):
        params = []
        for module, name, param in walk_module(self.module):
            if 'weight' in name and (with_norm or not isinstance(module, (_BatchNorm, nn.LayerNorm))):
                params.append(param)
        return params

    def bias_params(self, with_norm=False):
        params = []
        for module, name, param in walk_module(self.module):
            if 'bias' in name and (with_norm or not isinstance(module, (_BatchNorm, nn.LayerNorm))):
                params.append(param)
        return params

    def batchnorm_params(self):
        return [param for module, name, param in walk_module(self.module) if isinstance(module, _BatchNorm)]

    def layernorm_params(self):
        return [param for module, name, param in walk_module(self.module) if isinstance(module, nn.LayerNorm)]

    def norm_params(self):
        return [param for module, name, param in walk_module(self.module) if
                isinstance(module, (_BatchNorm, nn.LayerNorm))]

    @staticmethod
    def create_param_group(params, **kwargs):
        kwargs['params'] = params
        return kwargs
