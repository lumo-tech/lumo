from lumo import DatasetBuilder
from transformers.models.bert import BertTokenizer
import torch
import random

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dictionary = list(tokenizer.vocab.keys())


def get_dummy_sent():
    return ' '.join(random.sample(dictionary, random.randint(10, 20)))


data = [get_dummy_sent() for _ in range(500)]
ys = torch.randint(0, 10, (500,))

db = (
    DatasetBuilder()
        .add_input('xs', data)
        .add_input('ys', ys)
        .add_output('xs', 'xs')
        .add_output('ys', 'ys')
)

from lumo import CollateBase


class TokenizerCollate(CollateBase):
    def __init__(self, tokenizer: BertTokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def collate(self, sample_list):
        xs = [sample['xs'] for sample in sample_list]
        ys = [sample['ys'] for sample in sample_list]

        input_txt = self.tokenizer(xs, return_tensors='pt', padding=True)
        return {
            'xs': input_txt,
            'ys': ys
        }


loader = db.DataLoader(batch_size=10, collate_fn=TokenizerCollate(tokenizer))
for batch in loader:
    print(batch['xs'])
    break
