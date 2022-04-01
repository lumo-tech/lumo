- 记录实验，负责实验的组织、目录管理、元数据记录

```
from lumo_experiment import Experiment

Experiment('name')
```

Experiment 的意义：

- 为每个实验分配唯一存储空间（通过 exp_name/test_name ）
- 为每个实验保留回溯可能（通过 git）
- 为每个实验进行标注，方便(可能存在的)检索
- 其他状态动态记录