

# Configuration

lumo 提供了多级作用域的配置，这包括：

```
~/.lumorc.json -> user-level
<repo>/.lumorc.json -> repo-level, private
<repo>/.lumorc.public.json -> repo-level, public
```

所有的配置会在运行时加载到 `lumo.glob` 中，用以全局设置：

```
from lumo import glob

glob['xxx']
```

## 配置和超参数的区别

在 lumo 中，配置大多用于和实验完全无关，但跟电脑环境和 lumo 行为有关的内容，如数据集的存放路径、GitHub 的 access token 等。 `lumo` 中所有支持的可选行为都可以通过更改 glob 的配置进行控制，以下是目前支持变更的配置项：

| 配置项 | 描述 |
| --- | --- |
| github_access_token | 代替 exp.backup() 方法的 access_token 参数。 |
| exp_root | 一些初始路径之一。 |
| db_root | 一些初始路径之一。 |
| progress_root | 一些初始路径之一。 |
| metric_root | 一些初始路径之一。 |
| cache_dir | 一些初始路径之一。 |
| blob_root | 一些初始路径之一。 |
| timezone | 决定 lumo 使用的时区，默认为 'Asia/Shanghai'。 |
| TRAINER_LOGGER_STDIO | 控制 Logger 是否向标准输出流输出内容。 |
| dev_branch | 版本控制时代码快照的保存分支，默认为 'lumo_experiments'。 |
| HOOK_LOCKFILE | 控制加载 LockFile ExpHook 的行为。 |
| HOOK_RECORDABORT | 控制加载 RecordAbort ExpHook 的行为。 |
| HOOK_GITCOMMIT | 控制加载 GitCommit ExpHook 的行为。 |

