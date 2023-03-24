import warnings

import torch
from accelerate.accelerator import Accelerator as _Accelerator
from accelerate.data_loader import prepare_data_loader
from lumo.trainer.backend.base import Accelerator as Base


class Accelerator(_Accelerator):
    """
     Accelerator instance for distributed training (on multi-GPU, TPU) or mixed precision training.

     This Accelerator subclass is used for `Trainer`. The only difference is that
     the device of data will be controlled by `Trainer` rather than `Accelerator`.
     """

    def prepare_data_loader(self, data_loader, **kwargs):
        """
        Prepare the data loader.

        Args:
            data_loader: The data loader.
            **kwargs: Additional keyword arguments.

        Returns:
            The prepared data loader.
        """
        return prepare_data_loader(
            data_loader,
            None,  # None instead of self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=self.device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
        )


class HugAccelerator(Base):
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
