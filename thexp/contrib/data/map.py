from collections import defaultdict

import numpy as np
import torch


def create_id_dict(targets):
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    else:
        targets = np.array(targets)

    id_dict = defaultdict(list)
    for i, target in enumerate(targets):
        id_dict[target].append(i)

    return id_dict


