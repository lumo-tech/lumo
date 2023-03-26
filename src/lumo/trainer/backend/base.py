import torch
from lumo import DataLoaderSide
from lumo.proc.dist import gather
from torch import nn
from torch.utils.data import DataLoader


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
