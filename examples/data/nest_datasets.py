import random

from lumo import DatasetBuilder
from torchvision.transforms import transforms
import torch

# Create a mnist-like dummy dataset
db = (
    DatasetBuilder()
        .add_idx('id')
        .add_input("xs", torch.rand(500, 28, 28))
        .add_input("ys", torch.randint(0, 10, (500,)))
        .add_output("xs", "xs1", transforms.RandomHorizontalFlip())
        .add_output("xs", "xs2")
        .add_output("ys", "ys")
)


# Watch dataset structure
class SameClass:
    def __init__(self, db: DatasetBuilder):
        self.db = db
        ys = db.inputs['ys']
        cls_num = len(set(ys.tolist()))
        pos_cls = []
        for i in range(cls_num):
            pos_cls.append(torch.where(ys == i)[0])
        self.pos_cls = pos_cls

    def __call__(self, ys):
        index = random.choice(self.pos_cls[ys])
        return self.db[index]


pos_db = db.copy()
db.add_output('ys', 'pos', SameClass(pos_db))

sample = db[0]
print(sample['id'])
print(sample['ys'])
print(sample['pos']['id'])
print(sample['pos']['ys'])
