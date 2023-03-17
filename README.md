[中文](./README.ch.md)

# lumo

[![PyPI version](https://badge.fury.io/py/lumo.svg)](https://badge.fury.io/py/lumo)
![Python-Test](https://github.com/pytorch-lumo/lumo/actions/workflows/python-test.yml/badge.svg)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)
![Python-doc](./images/docstr_coverage_badge.svg)

`lumo` is a streamlined and efficient library that simplifies the management of all components required for experiments
and focuses on enhancing the experience of deep learning practitioners.

- **Experiment Management:** Allocates a unique path for each run, distinguishes and stores files for different types of
  experiments, and manages code snapshots via Git.
- **Parameter Management:** Provides more convenient parameter management than argparser based on fire.
- **Runtime Configuration:** Provides configuration management under multi-level scopes.
- **Visualization:** Provides an interactive Jupyter experiment management panel based
  on [Panel](https://panel.holoviz.org/index.html).
- Additional optimization for deep learning:
    - **Training:** Provides easily extendable training logic based on Trainer and provides comprehensive callback
      logic.
    - **Optimizer:** Integrated parameter and optimizer construction.
    - **Data:** Abstraction of dataset construction process, combination of multiple DataLoaders, etc.
    - **Distributed Training:** Also supports multiple training acceleration frameworks, unified abstraction, and easy
      switching at any time.
- More utilities...

# :book: Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)

# :cloud: Installation

Install the published and tested version:

```bash
pip install -U lumo
```

Or install the latest version from the dev1 branch:

```bash
pip install git+https://github.com/pytorch-lumo/lumo@dev1
```

The experiment panel depends on Panel, which needs to be installed separately:

```
pip install panel
```

# :book: Quick Start

Here are two classic scenarios:

## :small_orange_diamond: Embedding into Existing Projects

For existing projects, you can quickly embed Lumo by following these steps:

- Import Lumo and initialize Logger and Experiment:

```python
import random
from lumo import SimpleExperiment, Params, Logger, Meter, Record

logger = Logger()
exp = SimpleExperiment(exp_name='my_exp_a')
exp.start()
logger.add_log_dir(exp.mk_ipath())
```

- Initialize parameters:

```python
params = Params()
params.dataset = params.choice('cifar10', 'cifar100')
params.alpha = params.arange(default=1, left=0, right=10)
params.from_args()  # python3 train.py --dataset=cifar100 --alpha=0.2
print(params.to_dict())  # {"dataset": "cifar100", "alpha": 0.2}
```

- Record parameters and store information during training:

```python
exp.dump_info('params', params.to_dict())
print(exp.test_name)

params.to_yaml(exp.mk_ipath('params.yaml'))

for i in range(10):
    max_acc = exp.dump_metric('Acc', random.random(), cmp='max')
    logger.info(f'Max acc {max_acc}')

    ckpt_fn = exp.mk_bpath('checkpoints', f'model_{i}.ckpt')
    ...  # save code given ckpt_fn

record = Record()
for batch in range(10):
    m = Meter()
    m.mean.Lall = random.random()
    m.last.lr = batch
    record.record(m)
    logger.info(record)

exp.end()
```

## :small_orange_diamond: Building from Scratch

If you want to start a new deep learning experiment from scratch, you can use Lumo to accelerate your code development.
Below are examples of Lumo training at different scales:

one-fine training:

| Example                                     | CoLab | Lines of Code |
|---------------------------------------------|-------|---------------|
| [MNIST example](./examples/mnist.py)        |       | 118           |
| [MocoV2 trains CIFAR10](./examples/moco.py) |       | 284   |
| [Multi-GPU training ImageNet]()             |       ||

Experimental project:

| Project                                                                                                       | Description                            |
|-----------------------------------------------------------------------------------------------------------|-------------------------------|
| [image-classification](https://github.com/pytorch-lumo/image-classification)                              | Reproducible code for multiple papers with full supervision, semi-supervision, and self-supervision      |
| [emotion-recognition-in-coversation](https://github.com/pytorch-lumo/emotion-recognition-in-conversation) | Reproducible code for multiple papers on dialogue emotion classification and multimodal dialogue emotion classification |

## :small_orange_diamond: Visual Interface

In jupyter:

```python
from lumo import Watcher

w = Watcher()
df = w.load()
widget = w.panel(df)
widget.servable()
```

![Panel](./images/panel-example.png)

Manually filtered experiments for visualization:
![Panel](./images/panel-example2.png)

You can directly use the command line:

```
lumo board [--port, --address, --open]
```

## :small_orange_diamond: re-run

Experiment that failed due to certain reasons can be **re-run by using the unique experiment ID (test_name)** , extra
parameters can be **reassigned and replaced**.

```
lumo rerun 230313.030.57t --device=0
```

## :small_orange_diamond: backup

Backing up experiment information to a Github issue (based on PyGitHub):

```python
from lumo import Experiment, Watcher
from lumo import glob

glob[
    'github_access_token'] = 'ghp_*'  # Default value for `access_token`. It is recommended to store the access_token in the global configuration `~/.lumorc.json`.

w = Watcher()
df = w.load()

# Selecting a single experiment for backup
exp = Experiment.from_cache(df.iloc[0].to_dict())
issue = exp.backup('github', repo='pytorch-lumo/image-classification-private',
                   access_token='ghp_*',
                   update=True,  # If already backed up, overwrite the previous issue
                   labels=None,  # Optional labels
                   )
print(issue.number)

# Batch backup and add labels based on each experiment's parameters
issues = df.apply(
    lambda x: Experiment.from_cache(x.to_dict()).backup(..., labels=[x['params'].get('dataset', '')]),
    axis=1
)
```

![backup_github](./images/backup_github.png)

# :scroll: License

Distributed under the Apache License Version 2.0. See [LICENSE](./LICENSE) for more information.
