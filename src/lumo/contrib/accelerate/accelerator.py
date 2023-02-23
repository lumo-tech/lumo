from accelerate.accelerator import Accelerator as _Accelerator
from accelerate.data_loader import prepare_data_loader


class Accelerator(_Accelerator):
    """
    Accelerator instance for distributed training (on multi-GPU, TPU) or mixed precision training.

    This Accelerator subclass is use for `Trainer`, the only difference is that
    the device of data will be controlled by Trainer rather than Accelerator.
    """

    def prepare_data_loader(self, data_loader):
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
