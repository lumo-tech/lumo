#! https://www.zhihu.com/question/384519338/answer/1171163247

[lumo](https://github.com/lumo-tech/lumo) 是我打磨了 4 年的实验管理框架，提供了良好的实验管理细节功能，并针对深度学习的各个细节进行了优化。此外，在[图像分类](https://github.com/sailist/image-classification)和[多模态情感分类](https://github.com/sailist/emotion-recognition-in-conversation/)两个 track 下的实践已证明了将 lumo 用于实验管理的可行性。以下是 lumo 可以用于节省时间的内容：

# 减少代码改动和扩展成本

## 减少参数定义的复杂程度

基于 argparser 定义命令行参数存在几个问题：

- 实验设计中可能会频繁的改动入参，argparser 单行定义相对较长
- 缺少 typehint，，没有充分利用现代 IDE 的功能，导致无法快速的从使用定位到定义，同时也可能因为拼写错误等导致频繁的出错和更改

lumo 提供了针对 argparser 的平替方法 `Params`，将定义和使用融为一体：

```python
from lumo import Params
params = Params()
params.dataset = params.choice('cifar10', 'cifar100')
params.alpha = params.arange(default=1, left=0, right=10)
params.from_args()  # python3 train.py --dataset=cifar100 --alpha=0.2
print(params.to_dict())  # {"dataset": "cifar100", "alpha": 0.2}

print(params.dataset)
```

`Params` 同样还支持从文件导入配置：

```python
params.from_json(fn)
params.from_yaml(fn)
params.from_hydra(...)
```

## 减少数据集变更成本

lumo 提供了针对数据集构建的工具类 `DatasetBuilder`，基于有向无环图的思路，用流的方式构建数据集，减少了数据集变更带来的代码修改量。

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

## 基于最佳实践扩展代码

lumo 提供了在[图像分类](https://github.com/sailist/image-classification)和[多模态情感分类](https://github.com/sailist/emotion-recognition-in-conversation/) 两个领域的两个模板。模板内包括了对多篇已有论文的复现代码。代码合适的组织了数据、模型和算法的位置和逻辑，保证了实验数量和代码复杂度呈线性关系。

# 减少实验管理的复杂程度

在 [深度学习科研，如何高效进行代码和实验管理？](https://www.zhihu.com/question/269707221/answer/985429519) 中的回答介绍了 lumo 的实验管理思想：不覆盖的记录每次实验的 `[参数，代码，输出]`。lumo 提供了用于实验记录的 `Experiment` 类和用于实验管理的 `Watcher`。

`Experiment` 用于保证每一次实验的所有环境参数都被自动记录，代码会被自动提交，所有基于 `Experiment` 自动或手动记录的内容都会被存储在唯一标识对应的路径中：

```python
logger = Logger()
# 定义及使用，无需转换
exp = SimpleExperiment(exp_name='my_exp_a')  # 为每种实验手动定义唯一名称
exp.start() # 触发 git commit 并记录当前 commit 的 hex

# 手动记录参数
params.to_json(exp.mk_ipath('params.json'))

# 手动记录指标
exp.dump_metric('Acc', 0.1, cmp='max')

# 所有自动/手动记录的参数：
print(exp.properties)
{'agent': nan,
 'backup': {'23-03-17-003438': {'backend': 'github',
                                'number': 9,
                                'repo': '...'},
            },
 'exception': nan,
 'execute': {'cwd': '~/Documents/Python/lumo',
             'exec_argv': ['~/Documents/Python/lumo/a.py'],
             'exec_bin': '~/.pyenv/versions/3.9.16/bin/python3.9',
             'exec_file': '~/Documents/Python/lumo/a.py',
             'repo': '~/Documents/Python/lumo'},
 'exp_name': 'my_exp_a',
 'git': {'commit': '1014b6b5',
         'dep_hash': 'c93b8c4e340882f55cf0c8e125fa0203',
         'repo': '~/Documents/Python/lumo'},
 'hooks': {'Diary': {'loaded': True, 'msg': ''},
           'FinalReport': {'loaded': True, 'msg': ''},
           'GitCommit': {'loaded': True, 'msg': ''},
           'LastCmd': {'loaded': True, 'msg': ''},
           'LockFile': {'loaded': True, 'msg': ''},
           'RecordAbort': {'loaded': True, 'msg': ''}},
 'lock': {'accelerate': '0.16.0',
          'decorator': '5.1.1',
          'fire': '0.5.0',
          'hydra': '1.3.2',
          'joblib': '1.2.0',
          'lumo': '0.15.0',
          'numpy': '1.24.2',
          'omegaconf': '2.3.0',
          'psutil': '5.9.4',
          'torch': '1.13.1'},
 'note': 'This is a Note',
 'params': {'alpha': 1, 'dataset': 'cifar10'},
 'paths': {'blob_root': '~/.lumo/blob',
           'cache_root': '~/.lumo/cache',
           'info_root': '~/.lumo/experiments'},
 'pinfo': {'hash': '0af4b77497c85bc5b65ccbdd9ff4ca0f',
           'obj': {'argv': ['~/.pyenv/versions/3.9.16/bin/python3.9',
                            '~/Documents/Python/lumo/a.py'],
                   'pid': 63975,
                   'pname': 'python3.9',
                   'pstart': 1678898740.099484},
           'pid': 63975},
 'progress': {'end': '23-03-16-004542',
              'end_code': 0,
              'last_edit_time': '23-03-16-004542',
              'ratio': 1,
              'start': '23-03-16-004542',
              'update_from': None},
 'tags': [],
 'test_name': '230316.000.8ct',
 'trainer': nan}
```

> `Experiment` 还提供了备份到 github issue、打包等更多功能。

`Watcher` 保证了所有实验中记录的内容都可以被检索并以 `pandas.DataFrame` 的格式提供更加灵活的筛选方式：

```python
from lumo import Watcher

w = Watcher()
df = w.load() # pandas.DataFrame
print(df.shape)
(453, 18)
print(df.columns)
Index(['exp_name', 'test_name', 'paths', 'note', 'lock', 'params', 'execute',
       'trainer', 'hooks', 'progress', 'pinfo', 'tags', 'git', 'metrics',
       'exception', 'agent', 'backup', 'info_dir'],
      dtype='object')
```

`Watcher` 还提供了一个用于 jupyter 的 panel，可以更好的对部分实验进行观察：

![](./images/panel-example2.png)


lumo，你值得尝试：

[lumo](https://github.com/lumo-tech/lumo)