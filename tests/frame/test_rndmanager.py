from thexp import RndManager

import random
import torch.nn as nn
import torch
from torch.optim import SGD


from thexp import RndManager


def test_rndmanager():
    rnd = RndManager()
    lis = []
    rnd_lis = []

    for i in range(2):
        rnd.mark("train")
        data = torch.rand(5, 2)
        y = torch.tensor([0, 0, 0, 0, 0])
        model = nn.Linear(2, 2)

        sgd = SGD(model.parameters(), lr=0.01)
        logits = model(data)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        sgd.step()
        sgd.zero_grad()

        lis.append(list(model.parameters()))

        rnd.shuffle() # 默认设置的种子是time.time()的小数部份，理论上重复概率极低
        rnd_lis.append(random.random())

    a,b = lis
    for pa,pb in zip(a,b):
        assert (pa == pb).all()

    a,b = rnd_lis
    assert a != b