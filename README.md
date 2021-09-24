[English](https://github.com/sailist/lumo/blob/master/README.en.md)

# lumo

`lumo` 是一个轻量级的 Pytorch 深度学习实验框架。

## 设计理念

- 高内聚，低耦合。所有模块尽可能解耦，除 [Trainer]() 外，其余模块均可以单独使用。
    - 这使得 lumo 不要求完全依赖。你可以只在你的项目中使用数据加载、训练逻辑、参数配置、日志输出中的一个或几个模块。
- 约定大于配置。在没有规定配置的地方，均采用默认配置，当存在特殊需求时候，自定义配置即可。
    - 灵活大于封装。在封装的同时保留原始训练逻辑，方便非通用训练内容的定制。
- 显式优于隐式。这是 Python
  的核心原则 [PEP 20](https://www.python.org/dev/peps/pep-0020/)。引用 [Django](https://docs.djangoproject.com/en/3.2/misc/design-philosophies/)
  设计哲学: 除非必要，否则魔法(Magic)不应该发生。除非有足够的收益，魔法不应该被使用。即使添加了魔法，它的实现方式也不应该使学习者感到困惑。

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
from lumo import Params

params = Params()
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
