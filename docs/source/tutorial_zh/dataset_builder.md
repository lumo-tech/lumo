# DatasetBuilder

lumo 设计了 `~lumo.DatasetBuilder` 用于提供一个统一的构建数据集的接口。可以在大多数情况下，减少重复的数据集形态设计。

以 CIFAR10 数据集为例，如果需要数据集的图片输出两次增广，那么要么要更改 Dataset 类，要么重写 transform 函数：

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

在多个数据集都面临这种更改时候，这种重写的方式会消耗较多的精力。尤其是在输出形式还不确定，可能面临频繁更改的时候。

对此，lumo 通过 `DatasetBuilder` 提供了一个通用的流式的解决方案，你只需要按标准形式准备好原始数据，和标准的 one-to-one 的增广函数：

```python
...

source = CIFAR10()
transform1 = ...
transform2 = ...
```

随后，任意的输出形式均可以通过 `DatasetBuilder` 来定义

```python
from lumo import DatasetBuilder

ds = (
    DatasetBuilder()
        # 定义输入流
        .add_input('xs', source.data)
        .add_input('ys', source.targets)
        # 定义输出流
        .add_output('xs','xs1',transform1) 
        .add_output('xs','xs2',transform2)
        .add_output('ys','ys')
)

print(ds[0])
>>> {'xs1': ..., 'xs2': ..., "ys": ...}
```
