from distutils.log import Log
from statistics import mode
from lumo import TrainerParams, Meter, DatasetBuilder, Record, Logger
from lumo.trainer.rnd import RndManager
from torch import nn
import torch

params = TrainerParams()  # A sub class of `omegaconf.DictConfig`
params.optim = params.OPTIM.create_optim("SGD", lr=0.00000000001)
params.batch_size = 128
params.epoch = 10
params.from_args()


def add(x):
    return x * 2


db = (
    DatasetBuilder()
        .add_input("xs", torch.arange(-2500, 2500, dtype=torch.float).unsqueeze(1))
        .add_input("ys", torch.arange(-2500, 2500, dtype=torch.float), transform=add)
        .add_output("xs", "xs")
        .add_output("ys", "ys")
)

loader = db.DataLoader(batch_size=params.batch_size, shuffle=True)

RndManager().mark(0)

model = nn.Linear(1, 1, bias=False)
optim = params.optim.build(model.parameters())

logger = Logger()

record = Record()
for i in range(params.epoch):
    for batch in loader:
        meter = Meter()
        xs = batch["xs"]
        ys = batch["ys"]
        out = model(xs)
        print(xs[:5])
        print(out[:5].long())
        print(ys[:5])
        loss = ((out - ys) ** 2).mean()
        loss.backward()
        optim.step()

        meter.mean.loss = loss
        meter.sum.c = 1
        record.record(meter)
        logger.inline(record.agg())
    logger.newline()

test_data = -torch.arange(1000, dtype=torch.float).unsqueeze(1)
res = model(test_data)

test_ys = test_data * 2
print((res.long() - test_ys.long()).sum())
