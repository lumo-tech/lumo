# lumo

`lumo`：轻量、可扩展的 Pytorch 实验框架。

## 设计理念

- 细节由使用者掌控：lumo 只封装了外部逻辑，内部实现由使用者自行实现（或借助框架用更少的代码量实现）
- 模块解耦合：所有模块可以单独作为您现在使用的框架中的一个插件使用
- 约定大于配置：尽可能的减少配置产生的心智负担；当你使用深度学习中一些约定俗成的变量名时（如 Loss/loss/Lall/Acc），lumo 会格外的注意它们！

## Install

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
from lumo import Record, Meter, Logger




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

### 训练

lumo.Trainer 的主要功能：

- 集成方法回调、变量记录、日志输出、模型存储、随机种子、实验管理等内置其他模块
- 集成 [huggingface/accelerate](https://github.com/huggingface/accelerate) ，用于多卡训练

