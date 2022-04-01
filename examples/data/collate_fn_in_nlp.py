from lumo import DatasetBuilder, CollateBase
from torchvision.transforms import transforms
from transformers.models.bert import BertTokenizer
import torch


class TokenizerCollate(CollateBase):

    def before_collate(self, sample_list):
        return super().before_collate(sample_list)


db = (
    DatasetBuilder()
        .add_input('xs', torch.rand(500, 14, 14, 3))
        .add_input('ys', torch.randint(0, 10, (500,)))
        .add_output('xs', 'xs1', transforms.RandomHorizontalFlip())
        .add_output('xs', 'xs2')
        .add_output('ys', 'ys')
)
print(db)
sample = db[0]
print(sample.keys())
print(sample['xs1'].shape, sample['xs2'].shape, sample['ys'])

loader = db.DataLoader(batch_size=10)
print(len(loader))
loader.set_batch_count(1024)  # will repeat __iter__ methods until satisfy the assigned `batch_count`
print(len(loader))

for batch in loader:
    print(batch['xs1'].shape)
    print(batch['ys'].shape)
    break
