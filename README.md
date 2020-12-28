[English](https://github.com/sailist/thexp/blob/master/README.en.md)

# thexp

`thexp` 是一个开源项目，是一个主要用于**深度学习 Research** 的 Pytorch 实验框架。

关于该项目的文档可以查看 [Document](https://sailist.github.io/thexp/zh/)



> 当前版本 1.5，框架整体代码已经稳定，不会轻易修改，只会补充细节或额外功能。

## Features

`thexp` 针对两个问题设计框架代码：Research 过程中试验设计多，代码修改频繁；深度学习除核心代码外冗余代码多，存在大量可简化代码。

目前，`thexp` 包含了以下 Features

 - 大幅度简化包括 **超参数配置**、**数据集构建**、**模型保存**、**断点恢复**、**变量记录** 与 **日志输出** 等各方面的冗余代码。
 - 通过内置 Git 操作和 **随机种子管理**，用简单的命令行即可任意的 **重置（reset）** 和 **打包（archive）** 用该框架完成的任意实验。
 - 通过配合框架内内置的**实验模板**，您可以以**接近线性增加的代码复杂度**在你的课题中添加任意多个相关实验。
 - 框架的构建遵循 **约定优于配置** 的原则，越以符合框架约定的方式编写和运行实验代码，框架能够为您做的也就更多，您管理和分析所做实验的方式也就越简单。

> 本项目配合自动补全友好的 IDE 使用更佳，（如 Pycharm）。

## Install
```bash
pip install thexp
```

您也可以在 github clone 并手动安装：
```bash
git clone https://github.com/sailist/thexp

python setup.py install
```



### 测试

```
python -m pytest # or python3 -m pytest
```

> 目前只有一部分代码包含测试用例。


## Introduction

和其他 `pytorch` 工具或框架的目的不全相同，`thexp` 主要用于科研用途，其编写时的核心想法有两点：

1. 减少重复代码，所写即所用
2. 所有操作**可记录**，**可回溯**，**可分析**


接下来我们可以简单的看一下 `thexp` 能帮你完成什么，以及它是否能够让您满意。


您可以通过 [Tutorial](https://sailist.github.io/thexp/zh/tutorial/) 部份来了解该框架的基本使用；随后，在您有更具体的需求时， [Cookbook](https://sailist.github.io/thexp/zh/cookbook/) 部份的介绍会让您更好的了解该框架的细节。

您可以按照如下的顺序对本框架的功能进行学习和使用：

 - 了解使用频率最高的用于训练流程的 [超参数声明 (Params) ](https://sailist.github.io/thexp/zh/params) 、[变量记录 (Meter)](https://sailist.github.io/thexp/zh/meter) 、[日志输出 (Logger) ](https://sailist.github.io/thexp/zh/logger) 、[数据集整合 (DataBundler) ](https://sailist.github.io/thexp/zh/bundler) 和将其整合在一起的 [训练流程抽象 (Trainer) ](https://sailist.github.io/thexp/zh/trainer)
 - 了解用于对实验进行分析的 [全局配置 (Config) ](https://sailist.github.io/thexp/zh/exp#全局配置) 和 [实验管理 (Experiment) ](https://sailist.github.io/thexp/zh/exp)
 - 了解面向部份需求的 [随机种子管理 (RndManager)](https://sailist.github.io/thexp/zh/rnd) 、[数据集形式构建 (DatasetBuilder)](https://sailist.github.io/thexp/zh/builder)
 
 在对这些内容都有所了解后，您可以通过 [Cookbook](https://sailist.github.io/thexp/zh/cookbook/) 查看关于 [实验模板](https://sailist.github.io/thexp/zh/structure) 的使用和其他 [细节](https://sailist.github.io/thexp/zh/details)

您也可以通过另外一个项目[thexp-implement](https://github.com/thexp/thexp-implement) 来了解如何使用，该项目基于 `thexp` 提供的模板对一些论文进行了复现，并会在随后陆续添加一些模型的 best result.
 
## Examples

在正式开始前，你可以通过下面一些简单的例子来了解`thexp`中的一些非常好用的功能。


### 超参数声明
支持多级嵌套、定义即所用的 Params
```python 
from thexp import Params
params = Params()
params.batch_size = 128
params.from_args() # 从命令行读入

>>> python ap.py --optim.lr=0.001 --epoch=400 --dataset=cifar10 --k=12
```
### 变量记录

以及，在使用变量的同时完成记录变量，并以极少的代码代价更新记录变量的平均值，这极大的简化了变量的记录逻辑，该部分位于 [变量记录]([#变量记录](https://sailist.github.io/thexp/zh/meter))。
```python
from thexp import Meter,AvgMeter

am = AvgMeter() # 用于记录并输出平均值
for j in range(500):
    meter = Meter()
    meter.percent(meter.c_) # 以百分比的形式输出变量 'c'
    meter.a = 1
    meter.b = "2"
    meter.c = torch.rand(1)[0]

    meter.loss = loss_fn(...)
    meter.rand = torch.rand(2)
    meter.d = [4] # 任意格式都可以存储并记录
    meter.e = {5: "6"}

    am.update(meter) # 记录一次当前的记录值用于平均。会根据类型和声明自动判断哪些可以求平均值。
    print(am)
```

### 分析代码

实验日志查询，以及对结果进行对比并绘制相应曲线。该部分位于 [实验管理 (Experiment) ](https://sailist.github.io/thexp/zh/exp)


```python
from thexp import Q, C

testq = (
    Q.repos() # 获取所有的项目
        .exps()['sup'] # 首先获取所有项目中的所有实验，其次用['sup']筛选出实验命名为 `sup` 的实验
        .tests() # 获取筛选出的实验的所有 `test`（试验），
)
bd = (
    testq.success()  # 筛选出所有运行过的 test 中成功的（正常结束程序）
        .boards()  # 获取这一部分的 test 的 BoardQuery
)

print(bd.scalar_tags) # 查看有哪些可绘制的 tag

bd.parallel(C.meter.top1_test_,  # 绘制平行图，对比的变量有 记录变量中的 top1 准确率
            C.param["optim.args.lr"],  # 超参数中的学习率
            C.param.epoch)  # 超参数中运行的 epoch
```
> <img src="/img/query_parallel.png" alt="平行图">

以及绘制曲线图：
```python
bd.boards().line('top1_test_')
```
> <img src="/img/query_line.png" alt="线图">

## Contribute

我相信这是一个非常好的，能够对大多数深度学习科研工作者都有帮助的开源项目，但该项目仍然存在诸多不足，包括：

 - [Tutorial](https://sailist.github.io/thexp/zh/tutorial/) 和 [Cookbook](https://sailist.github.io/thexp/zh/cookbook/) 部份已经覆盖了大部分的使用说明。但由于开发者时间精力有限，因此本项目在很多细节的地方，**缺少更完善的使用指南**。
 - **测试未覆盖全部代码**。部份关键部份存在测试用例，此外的所有已修复的 bug 均是在作者使用过程中不断迭代人肉修复的 Bug ，在长达几个月的不断完善中，理论上大部分的功能不存在致命的Bug，但仍然可能存在诸多没有发现的问题。此外，在不同电脑上的兼容情况可能也没有办法得到很好的保证。对于这部分，欢迎在 [issus](https://github.com/sailist/thexp/issues) 页面提出。
 - **缺少项目开发经验**。这不是一个工程项目，由于精力有限，以及没有实习经验，因此我没有比较好的基于 git 的开发方法和分支管理办法，也没有采用什么工程化的开发流程，版本号也定义的一塌糊涂，但这部分对于使用者而言可能并不重要。

因此，如果该库对你有帮助，或者你在使用过程中发现了问题，或者你希望该库更加的完善，那么希望你也可以参与到贡献中。仍然是由于精力问题，我不太懂贡献流程，因此除了代码问题、bug修复等，如果有人能贡献如何贡献的流程，或者贡献有人贡献出的如何贡献的流程（套娃禁止），也是非常有帮助的。

为一个开源项目进行贡献是十分消耗精力的一件事，因此十分感谢参与贡献的朋友。

以上！最后祝大家炼丹愉快！

