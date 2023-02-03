from accelerate.utils import recursively_apply
import torch


def send_to_device(tensor, device, non_blocking=False):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to a given device.
        device (`torch.device`):
            The device to send the data to.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    if 'mps' in device.type:
        non_blocking = False

    def _send_to_device(t, device):
        return t.to(device, non_blocking=non_blocking)

    def _has_to_method(t):
        return hasattr(t, "to")

    return recursively_apply(_send_to_device, tensor, device, test_type=_has_to_method)
