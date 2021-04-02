"""

"""

import torch

from lumo.calculate import accuracy as acc


def test_classify():
    labels = torch.tensor([0, 1, 2, 3])
    preds = torch.tensor([[5, 4, 3, 2], [5, 4, 3, 2], [5, 4, 3, 2], [5, 4, 3, 2]], dtype=torch.float)
    total, res = acc.classify(preds, labels, topk=(1, 2, 3, 4))
    assert total == 4
    assert res[0] == 1 and res[1] == 2 and res[2] == 3 and res[3] == 4,str(res)
