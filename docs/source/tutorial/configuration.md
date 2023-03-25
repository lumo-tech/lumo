# Runtime Configuration and Params

## Params

`~lumo.Params`is used to specify the configuration required for the current experiment. In addition to defining parameters that support autocompletion, it also supports command-line parameters, inheritance, and reading from multiple configuration files.

The simplest usage is as follows:

```python
from lumo import Params

params = Params()
params.lr = 1e-3
params.dataset = 'cifar10'
params.from_args() # python main.py --dataset=cifar100

print(params.dataset)
>>> "cifar100"
```

Limit the value of parameters:

```python
params.dataset = params.choice('cifar10', 'cifar100')
print(params.dataset)
>>> "cifar10" # by default is the first value

params.dataset = "imagenet"
>>> raise BoundCheckError: value of param 'dataset' should in values ('cifar10',
'cifar100'), but got imagenet
```

Read from other locations:

```python
params.from_json("*.json")
params.from_yaml("*.yaml")
params.from_yaml("*.yml")
params.from_dict({})
```

`params.config`or`params.c`is a built-in reserved parameter. When the values of these two variables are strings and the path judgment is a yaml or json file or file list, the configuration is read from the corresponding position:

```json
# cfg.json
{
    "dataset": "cifar100"
}
```

```python
params.from_args(['--c','cfg.json'])
print(params.dataset)
>>> "cifar100"
```

## Configuration

`lumo` provides a multi-level configuration system, including three file locations:

```arduino
~/.lumorc.json -> user-level
<repo>/.lumorc.json -> repo-level, private
<repo>/.lumorc.public.json -> repo-level, public
```

All configurations are loaded into`lumo.glob`at runtime for global settings:

```css
from lumo import glob

glob['xxx']
```

## Difference between Configuration and Hyperparameters

In `lumo`, configurations are mostly used for non-experiment-related content that is related to the computer environment and `lumo` behavior, such as the location of the dataset, GitHub access tokens, etc. All supported optional behaviors in`lumo`can be controlled by modifying the configuration in`glob`. The following are the currently supported configurable items:

| Configuration | Description |
| --- | --- |
| github_access_token | Replaces the access_token parameter of the exp.backup() method. |
| exp_root | One of several initial paths. |
| db_root | One of several initial paths. |
| progress_root | One of several initial paths. |
| metric_root | One of several initial paths. |
| cache_dir | One of several initial paths. |
| blob_root | One of several initial paths. |
| timezone | Determines the timezone used by lumo. Default is 'Asia/Shanghai'. |
| TRAINER_LOGGER_STDIO | Controls whether the Logger outputs to the standard output stream. |
| dev_branch | The branch used for saving code snapshots during version control. Default is 'lumo_experiments'. |
| HOOK_LOCKFILE | Behavior control for loading LockFile ExpHook. |
| HOOK_RECORDABORT | Behavior control for loading RecordAbort ExpHook. |
| HOOK_GITCOMMIT | Behavior control for loading GitCommit ExpHook. |

