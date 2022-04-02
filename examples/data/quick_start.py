from lumo import DatasetBuilder
from torchvision.transforms import transforms
import torch

# Create a mnist-like dummy dataset
db = (
    DatasetBuilder()
    .add_input("xs", torch.rand(500, 28, 28))
    .add_input("ys", torch.randint(0, 10, (500,)))
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
sample = db[0]
print(sample.keys())
print(sample["xs1"].shape, sample["xs2"].shape, sample["ys"])

# Construct dataloader from dataset 
## db.DataLoader return a partial DataLoader object, where `dataset=db``
loader = db.DataLoader(batch_size=10)
print(len(loader)) # 500 / 10 = 50

# set Dataset
loader.set_batch_count(
    1024
)  # will repeat __iter__ methods until satisfy the assigned `batch_count`
print(len(loader))

for batch in loader:
    print(batch["xs1"].shape)
    print(batch["ys"].shape)
    break
