from typing import Optional, List, Union

import torch
from accelerate import Accelerator as HugAccelerator, DeepSpeedPlugin
from accelerate.kwargs_handlers import DistributedDataParallelKwargs, GradScalerKwargs, KwargsHandler
from accelerate.state import AcceleratorState as HugAcceleratorState, DistributedType
from accelerate.utils import RNGType
from packaging import version

from lumo.data.accelerator import prepare_data_loader


class AcceleratorState(HugAcceleratorState):

    def __init__(self, fp16: bool = None, cpu: bool = False, _from_accelerator: bool = False, device=None):
        super().__init__(fp16, cpu, _from_accelerator)
        self.cpu = cpu
        self.to_device(device)

    def to_device(self, device):
        """device can be changed only in single GPU mode"""
        if device is not None:
            if self.distributed_type == DistributedType.NO:
                self.device = torch.device(device)
                return True
        return False


class Accelerator(HugAccelerator):

    def __init__(self, device_placement: bool = True, split_batches: bool = False, fp16: bool = None,
                 mixed_precision: str = None, cpu: bool = False, deepspeed_plugin: DeepSpeedPlugin = None,
                 rng_types: Optional[List[Union[str, RNGType]]] = None, dispatch_batches: Optional[bool] = None,
                 kwargs_handlers: Optional[List[KwargsHandler]] = None,
                 device=None,
                 accelerator_state: HugAcceleratorState = None):
        super().__init__(device_placement=device_placement,
                         split_batches=split_batches, fp16=fp16, mixed_precision=mixed_precision, cpu=cpu,
                         deepspeed_plugin=deepspeed_plugin, rng_types=rng_types,
                         dispatch_batches=dispatch_batches, kwargs_handlers=kwargs_handlers)

        if accelerator_state is None:
            self.state = AcceleratorState(fp16=fp16, cpu=cpu, _from_accelerator=True, device=device)
        else:
            self.state = accelerator_state
            if device is not None:
                if accelerator_state.distributed_type == DistributedType.NO:
                    accelerator_state.device = torch.device(device)

        self.device_placement = device_placement
        self.split_batches = split_batches

        # Kwargs handlers
        self.ddp_handler = None
        self.scaler_handler = None
        if kwargs_handlers is not None:
            for handler in kwargs_handlers:
                assert isinstance(handler, KwargsHandler), f"Unsupported kwargs handler passed: {handler}."
                if isinstance(handler, DistributedDataParallelKwargs):
                    if self.ddp_handler is not None:
                        raise ValueError("You can only pass one `DistributedDataParallelKwargs` in `kwargs_handler`.")
                    else:
                        self.ddp_handler = handler
                elif isinstance(handler, GradScalerKwargs):
                    if self.scaler_handler is not None:
                        raise ValueError("You can only pass one `GradScalerKwargs` in `kwargs_handler`.")
                    else:
                        self.scaler_handler = handler

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        if self.state.use_fp16:
            self.native_amp = version.parse(torch.__version__) >= version.parse("1.6")
            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            self.scaler = torch.cuda.amp.GradScaler(**kwargs)

        # Internal references to the training objects
        self._optimizers = []

        # RNG Types
        if rng_types is None:
            self.rng_types = ["torch"] if version.parse(torch.__version__) <= version.parse("1.5.1") else ["generator"]

    def prepare_data_loader(self, data_loader):
        return prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=self.device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
        )
