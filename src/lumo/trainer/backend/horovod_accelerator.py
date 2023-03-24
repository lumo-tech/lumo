import torch
from lumo import DataLoaderSide
from lumo.proc.dist import gather
from torch import nn
from torch.utils.data import DataLoader
from .base import Accelerator


class Horovod(Accelerator):
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
        super().__init__(**kwargs)
        import horovod

