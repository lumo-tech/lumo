from lumo import DatasetBuilder
from torchvision.transforms import transforms
import torch

sup_db = (
    DatasetBuilder()
        .add_input('xs', torch.rand(50, 14, 14, 3))
        .add_input('ys', torch.randint(0, 10, (50,)))
        .add_output('xs', 'xs1', transforms.RandomHorizontalFlip())
        .add_output('xs', 'xs2')
        .add_output('ys', 'ys')

)

un_db = (
    DatasetBuilder()
        .add_input('xs', torch.rand(500, 14, 14, 3))
        .add_input('ys', torch.randint(0, 10, (500,)))
        .add_output('xs', 'xs1', transforms.RandomHorizontalFlip())
        .add_output('xs', 'xs2')
        .add_output('ys', 'ys')
)

sup_db.scale_to_size(len(un_db))

db = (
    DatasetBuilder()
        .add_input('sup', sup_db)
        .add_input('un', un_db)
        .add_output('sup', 'sup')
        .add_output('un', 'un')
)

print(db)

sample = db[0]

print(sample.keys())
loader = db.DataLoader(batch_size=10)
loader.set_batch_count(1024)  # will repeat __iter__ methods until satisfy the assigned `batch_count`
print(len(loader))

for batch in loader:
    sup, un = batch['sup'], batch['un']
    print(un['xs1'].shape)
    print(sup['ys'].shape)
    break
