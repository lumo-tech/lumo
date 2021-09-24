import torch


class ClassCenter:
    def __init__(self, nclasses, norm=True, moment=0.99) -> None:
        super().__init__()
        self.nclasses = nclasses
        # self._center = torch.rand(nclasses)
        self.norm = norm
        self._center = None

    def _check_center(self, logits):
        if self._center is None:
            self._center = torch.rand(self.nclasses, device=logits.device, dtype=logits.dtype)
            return True
        return False

    def update(self, logits, ys):
        first = self._check_center(logits)

        for cls in set(ys.tolist()):
            cls_mask = (ys == cls)
            if first:
                self._center[cls] = logits[cls_mask].mean(dim=0)
            else:
                self._center[cls] = self._center[cls] * 0.99 + logits[cls_mask].mean(dim=0) * 0.01

