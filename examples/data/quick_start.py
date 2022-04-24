from lumo import DatasetBuilder
from torchvision.transforms import transforms
import torch

# Create a mnist-like dummy dataset
db = (
    DatasetBuilder()
        .add_input("xs", torch.rand(500, 28, 28))
        .add_input("ys", torch.randint(0, 10, (500,)))
        .add_idx('id')
        .add_output("xs", "xs1", transforms.RandomHorizontalFlip())
        .add_output("xs", "xs2")
        .add_output("ys", "ys")
)
# Watch dataset structure
print(db)

# Notice: when sources have different length,
##   DatasetBuilder will become an IterableDataset,
##   which means it can only be itered.
# get dataset
db.chain()
sample = db[0]
print(sample)
print(sample.keys())
print(sample["xs1"].shape, sample["xs2"].shape, sample["ys"])

# Construct dataloader from dataset
## db.DataLoader return a partial DataLoader object, where `dataset=db``
loader = db.DataLoader(batch_size=10)
print(len(loader))  # 500 / 10 = 50

# set Dataset
loader.set_batch_count(
    1024
)  # will repeat __iter__ methods until satisfy the assigned `batch_count`
print(len(loader))

for batch in loader:
    print(batch["xs1"].shape)
    print(batch["ys"].shape)
    break

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        self.xs = torch.rand(500, 28, 28)
        self.ys = torch.randint(0, 10, (500,))
        self.xs1_transform = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        return {
            'xs1': self.xs1_transform(self.xs[index]),
            'xs2': self.xs[index],
            'ys': self.ys[index],
        }
