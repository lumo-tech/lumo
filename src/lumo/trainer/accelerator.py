import warnings
from torch import nn
import torch
from torch import distributed
from torch.utils.data import DataLoader
from lumo.data.loader import DataLoaderSide


class Accelerator:
    def __init__(self, **kwargs):
        self._prop = kwargs

    @property
    def device(self):
        return self._prop.get('device', None)

    def set_device(self, device):
        assert isinstance(device, torch.device)
        self._prop['device'] = device

    def prepare_data_loader(self, dataloader):
        return dataloader

    def prepare_model(self, model: torch.nn.Module):
        return model.to(self.device)

    def prepare_optimizer(self, optimizer: torch.optim.Optimizer):
        return optimizer

    def unwrap_model(self, model):
        return model

    def prepare(self, *args):
        res = []
        for item in args:
            if isinstance(item, nn.Module):
                res.append(self.prepare_model(item))
            elif isinstance(item, (DataLoader, DataLoaderSide)):
                res.append(self.prepare_data_loader(item))
            elif isinstance(item, torch.optim.Optimizer):
                res.append(self.prepare_optimizer(item))
            else:
                raise NotImplementedError()
        return res

    def wait_for_everyone(self):
        torch.distributed.barrier()

    def gather(self, tensor: torch.Tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        distributed.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)


class HugAccelerator(Accelerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .backend.accelerator import Accelerator
        self._backbone = Accelerator()

    @property
    def device(self):
        return self._backbone.device

    def set_device(self, device: torch.device):
        assert isinstance(device, torch.device)
        self._backbone.state.device = device

    def prepare_data_loader(self, loader):
        from accelerate.data_loader import DataLoaderShard, DataLoaderDispatcher
        if isinstance(loader, (DataLoaderShard, DataLoaderDispatcher)):
            warnings.warn('Duplicated prepare a same DataLoader twice, check your code.')
            return loader
        return self._backbone.prepare_data_loader(loader)

    def prepare_model(self, model):
        return self._backbone.prepare_model(model)

    def prepare_optimizer(self, optimizer):
        return self._backbone.prepare_optimizer(optimizer)

    def unwrap_model(self, model):
        return self._backbone.unwrap_model(model)

    def prepare(self, *args):
        return self._backbone.prepare(*args)

    def wait_for_everyone(self):
        self._backbone.wait_for_everyone()

    def gather(self, tensor):
        return self._backbone.gather(tensor)


register = {

    'none': Accelerator,
    'accelerator': HugAccelerator,
    'deepspeed': None,
    'horovod': None,
}


def get_accelerator(name: str, **kwargs):
    assert name in register, ', '.join(register.keys())
    return register[name](**kwargs)
