from torch import nn
import torch

a = torch.zeros(20, 20)
a[:1].rand_()

torch.rand_like()

print(a[:1])