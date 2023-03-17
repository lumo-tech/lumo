import warnings
from torch import nn
import torch
from torch import distributed
from torch.utils.data import DataLoader
from lumo.data.loader import DataLoaderSide
from lumo.proc.dist import gather


class Accelerator:
    """
    A class to define the interface for various types of accelerator.

    Attributes:
        _prop (dict): A dictionary of keyword arguments.

    Methods:
        device: A property method to get the device.
        set_device: A method to set the device.
        prepare_data_loader: A method to prepare the data loader.
        prepare_model: A method to prepare the model.
        prepare_optimizer: A method to prepare the optimizer.
        unwrap_model: A method to unwrap the model.
        prepare: A method to prepare the inputs for training.
        wait_for_everyone: A method to wait for all processes to synchronize.
        gather: A method to gather the tensor data.
        backward: A method to compute the gradients using backpropagation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the class with a dictionary of keyword arguments.
        """
        self._prop = kwargs

    @property
    def device(self) -> torch.device:
        """
        Get the device.
        """
        return self._prop.get('device', None)

    def set_device(self, device: torch.device):
        """
        Set the device.

        Args:
            device (torch.device): The device to be set.
        """
        assert isinstance(device, torch.device)
        self._prop['device'] = device

    def prepare_data_loader(self, dataloader):
        """
        Prepare the data loader.

        Args:
            dataloader: The data loader.

        Returns:
            The prepared data loader.
        """
        return dataloader

    def prepare_model(self, model: torch.nn.Module):
        """
        Prepare the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            The prepared model.
        """
        return model.to(self.device)

    def prepare_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Prepare the optimizer.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            The prepared optimizer.
        """
        return optimizer

    def unwrap_model(self, model):
        """
        Unwrap the model.

        Args:
            model: The model.

        Returns:
            The unwrapped model.
        """
        return model

    def prepare(self, *args):
        """
        Prepare the inputs for training.

        Args:
            *args: The inputs.

        Returns:
            The prepared inputs.
        """
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
        """
        Wait for all processes to synchronize.
        """
        torch.distributed.barrier()

    def gather(self, tensor: torch.Tensor):
        """
        Gather the tensor data.

        Args:
            tensor (torch.Tensor): The tensor to be gathered.

        Returns:
            The gathered tensor data.
        """
        return gather(tensor)

    def backward(self, loss: torch.Tensor, **kwargs):
        """
        Compute the gradients using backpropagation.

        Args:
            loss (torch.Tensor): The loss tensor.
            **kwargs: The additional keyword arguments.
        """
        loss.backward(**kwargs)

class HugAccelerator(Accelerator):
    """
    A class to define the interface for Hugging Face accelerator.

    Methods:
        set_device: A method to set the device.
        prepare_data_loader: A method to prepare the data loader.
        prepare_model: A method to prepare the model.
        prepare_optimizer: A method to prepare the optimizer.
        unwrap_model: A method to unwrap the model.
        prepare: A method to prepare the inputs for training.
        wait_for_everyone: A method to wait for all processes to synchronize.
        gather: A method to gather the tensor data.
        backward: A method to compute the gradients using backpropagation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the class with a dictionary of keyword arguments.
        """
        super().__init__(**kwargs)
        from .backend.accelerator import Accelerator
        self._backbone = Accelerator()

    @property
    def device(self):
        """
        Get the device.
        """
        return self._backbone.device

    def set_device(self, device: torch.device):
        """
        Set the device.

        Args:
            device (torch.device): The device to be set.
        """
        assert isinstance(device, torch.device)
        self._backbone.state.device = device

    def prepare_data_loader(self, loader):
        """
        Prepare the data loader.

        Args:
            loader: The data loader.

        Returns:
            The prepared data loader.
        """
        from accelerate.data_loader import DataLoaderShard, DataLoaderDispatcher
        if isinstance(loader, (DataLoaderShard, DataLoaderDispatcher)):
            warnings.warn('Duplicated prepare a same DataLoader twice, check your code.')
            return loader
        return self._backbone.prepare_data_loader(loader)

    def prepare_model(self, model):
        """
        Prepare the model.

        Args:
            model: The model.

        Returns:
            The prepared model.
        """
        return self._backbone.prepare_model(model)

    def prepare_optimizer(self, optimizer):
        """
        Prepare the optimizer.

        Args:
            optimizer: The optimizer.

        Returns:
            The prepared optimizer.
        """
        return self._backbone.prepare_optimizer(optimizer)

    def unwrap_model(self, model):
        """
        Unwrap the model.

        Args:
            model: The model.

        Returns:
            The unwrapped model.
        """
        return self._backbone.unwrap_model(model)

    def prepare(self, *args):
        """
        Prepare the inputs for training.

        Args:
            *args: The inputs.

        Returns:
            The prepared inputs.
        """
        return self._backbone.prepare(*args)

    def wait_for_everyone(self):
        """
        Wait for all processes to synchronize.
        """
        self._backbone.wait_for_everyone()

    def gather(self, tensor):
        """
        Gather the tensor data.

        Args:
            tensor: The tensor to be gathered.

        Returns:
            The gathered tensor data.
        """
        return self._backbone.gather(tensor)

    def backward(self, loss: torch.Tensor, **kwargs):
        """
        Compute the gradients using backpropagation.

        Args:
            loss (torch.Tensor): The loss tensor.
            **kwargs: The additional keyword arguments.
        """
        self._backbone.backward(loss, **kwargs)


register = {

    'none': Accelerator,
    'accelerator': HugAccelerator,
    'deepspeed': None,
    'horovod': None,
}


def get_accelerator(name: str, **kwargs) -> Accelerator:
    """Get the accelerator for the specified name."""
    assert name in register, ', '.join(register.keys())
    return register[name](**kwargs)
