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

```
github_access_token -> 代替 exp.backup() 方法的 access_token 参数

一些初始路径
exp_root
db_root
progress_root
metric_root
cache_dir
blob_root

timezone -> 决定 lumo 使用的 timezone，默认是 'Asia/Shanghai'

TRAINER_LOGGER_STDIO -> Logger 是否向标准输出流输出内容。

dev_branch -> 版本控制时代码快照的保存分支，默认是 lumo_experiments


# ExpHook 是否加载的行为控制，该配置项没有hard code，而是通过检查 `f"HOOK_{ExpHook.__class__.__name__.upper()}"` 来判断的，如：
HOOK_LOCKFILE -> lumo.exp.exphook.LockFile
HOOK_RECORDABORT -> lumo.exp.exphook.RecordAbort
HOOK_GITCOMMIT -> lumo.exp.exphook.GitCommit
```
