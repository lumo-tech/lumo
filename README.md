[English](https://github.com/sailist/lumo/blob/master/README.en.md)

# lumo

`lumo` 是一个轻量级的 Pytorch 深度学习实验框架。

## Install

```bash
pip install lumo
```

。

## Quick start

### 参数控制

```python
from lumo import Params

params = Params()
params.epoch = 20
params.optim = params.OPTIM.create_optim('Adam', lr=0.0001, weight_decay=4e-5)
params.dataset = params.choice('cifar10', 'cifar100')

params.from_args()
```

### 训练

通过 Trainer 完成：

- 内置方法回调、变量记录、日志输出、模型存储、随机种子、实验管理等模块
- 集成 [huggingface/accelerate](https://github.com/huggingface/accelerate) ，随意切换单机 / 集群

[pytorch ImageNet example]() vs. [lumo ImageNet example]()

> 变量记录、日志输出、模型存储、随机种子、实验管理 等各模块均已解耦，可单独使用

### 通过流的方式搭建数据集

```python
from lumo import DatasetBuilder, DataModule, K


class Key(K):
    pass


ds = (
    DatasetBuilder()

)
```
