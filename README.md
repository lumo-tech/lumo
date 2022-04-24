# lumo

`lumo`：轻量、可扩展的 Pytorch 实验框架。

## 设计理念

- 细节由使用者掌控：lumo 只封装了外部逻辑，内部实现由使用者自行实现（或借助框架用更少的代码量实现）
- 模块解耦合：所有模块可以单独作为您现在使用的框架中的一个插件使用
- 约定大约配置：尽可能的减少配置产生的心智负担；当你使用深度学习中一些约定俗成的变量名时（如 Loss/loss/Lall/Acc），lumo 会格外的注意它们！

## Install

```bash
pip install lumo
```

或手动安装最新版本

```bash
pip install git+https://github.com/sailist/lumo
```

## Quick start

### 参数控制

```python
from lumo import Params, TrainerParams

params = TrainerParams()
params.epoch = 20
params.optim = params.OPTIM.create_optim('Adam', lr=0.0001, weight_decay=4e-5)
params.dataset = params.choice('cifar10', 'cifar100')

params.from_args()
params.to_json('./config.json')
```

### 变量&日志记录

```pycon
from lumo import AvgMeter, Meter, Logger



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
