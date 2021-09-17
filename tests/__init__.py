from lumo.kit.meter import AvgItem
import torch

import numpy as np

a = AvgItem(torch.rand(4),'sum')
a.update([np.random.rand(4)])