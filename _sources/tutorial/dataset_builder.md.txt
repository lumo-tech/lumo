# Build your Dataset Easily

lumo has designed`~lumo.DatasetBuilder`to provide a unified interface for constructing datasets. This can greatly reduce repetitive dataset design in most cases.

Taking the CIFAR10 dataset as an example, if the dataset requires images to be output with two augmentations, either the Dataset class needs to be modified or the transform function needs to be rewritten:

```python
class MyCIFAR(CIFAR10):
    ...

    def __getitem(self,index):
        sample = self.data[index]
        label = self.target[index]
        return self.transform1(sample), self.transform2(sample), label


# or

def two_transform(sample):
    ...
    return transform1(sample), transform2(sample)
```

When facing such changes in multiple datasets, this rewriting method can be time-consuming. Especially when the output format is not yet certain and may be subject to frequent changes.

To solve this, lumo provides a universal and streaming solution through`DatasetBuilder`. You only need to prepare the raw data in the standard format and the standard one-to-one augmentation functions:

```python
...

source = CIFAR10()
transform1 = ...
transform2 = ...
```

Then, any output format can be defined through`DatasetBuilder`:

```python
from lumo import DatasetBuilder

ds = (
    DatasetBuilder()
        # Define input stream
        .add_input('xs', source.data)
        .add_input('ys', source.targets)
        # Define output stream
        .add_output('xs','xs1',transform1)
        .add_output('xs','xs2',transform2)
        .add_output('ys','ys')
)

print(ds[0])
>>> {'xs1': ..., 'xs2': ..., "ys": ...}
```
