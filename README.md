# lumo

`lumo`：轻量、可扩展的 Pytorch 实验框架。

## 设计理念

- 细节由使用者掌控：lumo 只封装了外部逻辑，内部实现由使用者自行实现（或借助框架用更少的代码量实现）
- 模块解耦合：所有模块可以单独作为您现在使用的框架中的一个插件使用
- 约定大于配置：尽可能的减少配置产生的心智负担；当你使用深度学习中一些约定俗成的变量名时（如 Loss/loss/Lall/Acc），lumo 会格外的注意它们！

## 安装

```bash
pip install lumo
```

或手动安装最新版本

```bash
pip install git+https://github.com/sailist/lumo
```

## 快速使用指南

大部分模块都做了解耦，所以可以拆出来单独使用

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

```python
import random
import time

from lumo import Record, Meter, Logger

log = Logger()

record = Record()
for idx in range(256):
    meter = Meter()
    meter.last.i = idx
    meter.sum.acc = idx
    meter.mean.loss = random.random()

    record.record(meter)
    log.inline(record)
    time.sleep(0.5)
    if idx % 50 == 0:
        log.newline()
        record.clear()
log.info(record)
```

### 实验代码记录（通过 git）

```python
from lumo import SimpleExperiment
from lumo import Params

pm = Params()
pm.module = 'example'
pm.from_args()

exp = SimpleExperiment(pm.module)
exp.start()
# use `Experiment` to manage path
fn = exp.blob_file('checkpoint.pt')
with open(fn, 'w') as w:
    w.write('write big data in blob file')

fn = exp.test_file('params.json')
pm.to_json(fn)

print(exp.test_root)
print(exp.get_prop('git'))  # see git commit history
exp.end()
```

### 数据集控制

顶好用的 Dataset 构建类。

```python
from lumo import DatasetBuilder
from torchvision.transforms import transforms
import torch

# Create a mnist-like dummy dataset
db = (
    DatasetBuilder()
    .add_input("xs", torch.rand(500, 28, 28))
    .add_input("ys", torch.randint(0, 10, (500,)))
    .add_idx('id')
    .add_output("xs", "xs1", transforms.RandomHorizontalFlip())
    .add_output("xs", "xs2")
    .add_output("ys", "ys")
)
# Watch dataset structure
print(db)
# Builder(flow={'::idx::': ['id'], 'xs': ['xs1', 'xs2'], 'ys': ['ys']}, sized=True, size=500, iterable=True)

print(db[0])
# dict_keys(['id', 'xs1', 'xs2', 'ys'])
```


进一步使用查看 [daatset.py](./examples/data/quick_start.py)

### 训练

lumo.Trainer 的主要功能：

- 集成方法回调、变量记录、日志输出、模型存储、随机种子、实验管理等内置其他模块
- 集成 [huggingface/accelerate](https://github.com/huggingface/accelerate) ，用于多卡训练

