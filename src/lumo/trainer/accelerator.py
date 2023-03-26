from lumo.trainer.backend.accelerator import HugAccelerator
from lumo.trainer.backend.base import Accelerator
from lumo.trainer.backend.horovod_accelerator import Horovod
from lumo.trainer.backend.original import TorchDist

register = {
    'none': Accelerator,
    'original': TorchDist,
    'accelerator': HugAccelerator,
    'horovod': Horovod,
}


def get_accelerator(name: str, **kwargs) -> Accelerator:
    """Get the accelerator for the specified name."""
    assert name in register, ', '.join(register.keys())
    return register[name](**kwargs)
