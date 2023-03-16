# lumo

[![PyPI version](https://badge.fury.io/py/lumo.svg)](https://badge.fury.io/py/lumo)
![Python-Test](https://github.com/pytorch-lumo/lumo/actions/workflows/python-test.yml/badge.svg)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)
![Python-doc](./images/docstr_coverage_badge.svg)

`lumo` 是一个精简高效的库，简化了实验所需的所有组件的管理，并特别关注增强深度学习实践者的体验。

- 实验管理：: 为每次运行分配唯一路径，区分不同类型的文件并存储；并通过 git 管理代码快照。
- 参数管理：基于 fire 提供比 argparser 更便捷的参数管理
- 运行时配置：提供多级作用域下的配置管理
- 可视化：基于 [Panel](https://panel.holoviz.org/index.html) 提供可交互的 jupyter 实验管理面板
- 为深度学习提供额外的优化
    - 训练：基于 Trainer 提供可任意扩展的训练逻辑简化，并提供完善的回调逻辑
    - 优化器：参数与优化器构建一体化
    - 数据: 数据集构建流程抽象、组合多个 DataLoader、...
    - 分布式训练：同样支持多种训练加速框架，统一抽象，方便随时切换
- 更多工具类...

# :book: 目录

- [安装](#安装)
- [快速开始](#快速开始)

# :cloud: 安装

安装已发布的通过了所有测试的版本

```bash
pip install -U lumo
```

或从 dev1 分支安装最新版本：

```bash
pip install git+https://github.com/pytorch-lumo/lumo
```

实验面板依赖于 panel，需要额外安装：

```
pip install panel
```

# :book: 快速开始

以下是两个经典场景：

## :small_orange_diamond: 已有项目嵌入

对已有项目，可以通过以下方式快速嵌入

 - 引入
```python
import random
from lumo import SimpleExperiment, Params, Logger, Meter, Record
```

 - 初始化 Logger 和 Experiment
```python
logger = Logger()
# 定义及使用，无需转换
exp = SimpleExperiment(exp_name='my_exp_a')  # 为每种实验手动定义唯一名称
exp.start()
logger.add_log_dir(exp.mk_ipath())
```

 - 初始化参数
```python
# 替换基于 argparse 等的参数定义方法
params = Params()
params.dataset = params.choice('cifar10', 'cifar100')
params.alpha = params.arange(default=1, left=0, right=10)
params.from_args()  # python3 train.py --dataset=cifar100 --alpha=0.2
print(params.to_dict())  # {"dataset": "cifar100", "alpha": 0.2}
```

 - 在训练过程中记录参数、存储信息
```python
# 记录实验参数
exp.dump_info('params', params.to_dict())
print(exp.test_name)  # 为每次实验自动分配唯一名称

# 基于命名空间提供本次实验的唯一路径
# 元数据和二进制大文件分离，方便清理
params.to_yaml(exp.mk_ipath('params.yaml'))

for i in range(10):
    # 记录实验指标
    max_acc = exp.dump_metric('Acc', random.random(), cmp='max')
    logger.info(f'Max acc {max_acc}')

    # 存储大文件/二进制文件（如模型权重）
    ckpt_fn = exp.mk_bpath('checkpoints', f'model_{i}.ckpt')
    ...  # 保存代码 given ckpt_fn

record = Record()
for batch in range(10):
    m = Meter()
    m.mean.Lall = random.random()
    m.last.lr = batch
    record.record(m)
    logger.info(record)

# 主动结束实验，补充元信息。也可以在进程结束后由 hook 自动结束，支持针对异常的记录
exp.end()
```

## :small_orange_diamond: 从零开始

如果从新开始一个深度学习实验，那么可以使用 lumo 全方位的加速代码的构建，下面提供了多个不同规模下使用 lumo 训练的示例：

单文件：

| 示例                                     | CoLab | 代码行数 |
|----------------------------------------|-------|------|
| [MNIST 示例](./examples/mnist.py)        |       | 118  |
| [MocoV2 训练 CIFAR10](./examples/moco.py) || 284   |
| [多卡训练 ImageNet]()                      |||

实验项目：

| 项目                                                                                                        | 说明                            |
|-----------------------------------------------------------------------------------------------------------|-------------------------------|
| [image-classification](https://github.com/pytorch-lumo/image-classification)                              | 集成了全监督、半监督、自监督的多个论文的复现代码      |
| [emotion-recognition-in-coversation](https://github.com/pytorch-lumo/emotion-recognition-in-conversation) | 集成了对话情感分类、多模态对话情感分类的多个论文的复现代码 |

## :small_orange_diamond: 可视化界面

在 jupyter 中：

```python
from lumo import Watcher

w = Watcher()
df = w.load()
widget = w.panel(df)
widget.servable()
```

![Panel](./images/panel-example.png)

可视化手动筛选后的实验：
![Panel](./images/panel-example2.png)

可以直接使用命令行打开页面查看当前所有实验：

```
lumo board [--port, --address, --open]
```

# More

# :pencil: Acknowledge

从 2020 年维护至今。感谢 lumo 陪我见证我的学术生涯。

# :scroll: License

采用 [GNU General Public License 3.0 协议](./LICENSE)。

