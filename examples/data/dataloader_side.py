from lumo import DatasetBuilder, DataLoaderSide
from torchvision.transforms import transforms
import torch

sup_db = (
    DatasetBuilder()
    .add_input("xs", torch.rand(50, 14, 14))
    .add_input("ys", torch.randint(0, 10, (50,)))
    .add_output("xs", "xs1", transforms.RandomHorizontalFlip())
    .add_output("xs", "xs2")
    .add_output("ys", "ys")
)

un_db = (
    DatasetBuilder()
    .add_input("xs", torch.rand(500, 14, 14))
    .add_input("ys", torch.randint(0, 10, (500,)))
    .add_output("xs", "xs1", transforms.RandomHorizontalFlip())
    .add_output("xs", "xs2")
    .add_output("ys", "ys")
)

sup_dl = sup_db.DataLoader(batch_size=10)
un_dl = un_db.DataLoader(batch_size=10 * 7)
un_dl.set_batch_count(1024)

loader = (
    DataLoaderSide()
    .add("sup", sup_dl, cycle=True)
    .add("un", un_dl)
    .zip()  # batches are feeded as a dict.
    # .chain() # as a list (in added order).
)

# repeat __iter__ methods until satisfy the assigned `batch_count`
print(len(loader))

for batch in loader:
    sup, un = batch["sup"], batch["un"]
    # sup, un = batch # when .chain() are uncommented.
    print(un["xs1"].shape)
    print(sup["ys"].shape)
    break
