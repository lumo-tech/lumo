"""

"""
import sys
sys.path.insert(0,"../")
from lumo import __version__
import time
print(__version__)

import random
# from lumo.utils import random as rnd
import torch.nn as nn
import torch
from torch.optim import SGD


from lumo import RndManager
rnd = RndManager()

for i in range(2):
    rnd.mark("train")
    # rnd.fix_seed(1)
    data = torch.rand(5, 2)
    y = torch.tensor([0, 0, 0, 0, 0])
    model = nn.Linear(2, 2)

    sgd = SGD(model.parameters(), lr=0.01)
    logits = model(data)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    sgd.step()
    sgd.zero_grad()

    print(list(model.parameters()))

    rnd.shuffle()
    print(random.random())
