# lumo

[![PyPI version](https://badge.fury.io/py/lumo.svg)](https://badge.fury.io/py/lumo)
![Python-Test](https://github.com/pytorch-lumo/lumo/actions/workflows/python-test.yml/badge.svg)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)

`lumo` is a light-weight library to help construct your experiment code, record your experiment results, especially in the field of deep learning.


## Features

`lumo` is designed for reducing difficulty of the frequent code modification in experiments and simplify the redundant code.

At present, `lumo` has these features:

 - Simplest code for **Hyperparameter Configuration**、**Dataset Building**、**Module Checkpoint**、**Meter and Log**.
 - Include Git support and random seed management. You can **reset** and **archive** and **reimplement your experiments** by using simple console command.
 - Include a **deep learning experiment code templete**. You can add any experiments with linearly increasing code complexity by using it.
 - The framework follows the design paradigm of **convention over configuration**, the more you follow the convention, the more the framework will do for you.

> Better use Pycharm.

See [document](https://sailist.github.io/lumo/) for details. 



## Install
```bash
pip install lumo
```

or 

```bash
git clone https://github.com/sailist/lumo

python setup.py install
```

### test

```
python -m pytest # or python3 -m pytest
```

> Only a part of code have unit test.


## Requirements

 - install lumo will automatically install three light other libraries: [fire](https://github.com/google/python-fire), [psutil](https://github.com/giampaolo/psutil), [joblib](https://github.com/joblib/joblib).
 - lumo has mandatory dependencies on `pytorch`, `pandas` and `numpy`, you should manully install these before using lumo since they are usually common-used.
 - lumo has an optional dependency on `GitPython` as a plugin to execute git command, you can run `pip install GitPython` to install it.

```shell
pip install pandas numpy GitPython
```
and then see [pytorch](https://pytorch.org/) to install torch. 



## Introduction

Unlike other pytorch tools, `lumo` mainly designed for research, there are two core idea of it:

1. Reduce repetition of your code.
2. Make all operations **recordable**, **resumable**, **analyzable**.


Your can click [Tutorial](https://sailist.github.io/lumo/tutorial/) to learn the basic use of this framework. After that, you can view [Cookbook](https://sailist.github.io/lumo/cookbook/) to see some details of this library.

A suggested learning order may be：

 - Learn highly frequency used module: [Define hyperparameter(Params)](https://sailist.github.io/lumo/params)、[Record variable(Meter)](https://sailist.github.io/lumo/meter)、[Log(Logger)](/lumo/logger)、[Reshape your dataloader(DataBundler)](https://sailist.github.io/lumo/bundler) and their aggregation [Trainer](https://sailist.github.io/lumo/trainer).
 - Learn how to manage/analyse your experiment by [Config](https://sailist.github.io/lumo/exp) and [Experiment](https://sailist.github.io/lumo/exp)
 - Learn how to simple manage random seed by [RndManager](https://sailist.github.io/lumo/rnd) and to create your dataset elegantly by [DatasetBuilder](https://sailist.github.io/lumo/builder)

After learning above contents, you can view [Cookbook](https://sailist.github.io/lumo/cookbook/) to learn the use of [tempelet code](https://sailist.github.io/lumo/structure) and other [details](https://sailist.github.io/lumo/details).

You can also view another repository [lumo-implement](https://github.com/lumo/lumo-implement) to see a bigger example, it will continuously reimplement papers I interested by using the templete provided in `lumo`. 

## Examples

Before start, maybe you'd like to see some simple examples to learn what can `lumo` do.

### Define hyperparameters
By use `lumo.frame.Params`, you can define hyperparameters simply. See [Params](https://sailist.github.io/lumo/params) for details.
```python 
from lumo import Params
params = Params()
params.batch_size = 128
params.from_args() # from command args

>>> python ap.py --optim.lr=0.001 --epoch=400 --dataset=cifar10 --k=12
```
### Record variable

By using `lumo.frame.Meter`, you can record variable and update its average value with as little code as possible. See [Meter](https://sailist.github.io/lumo/meter) for details.

```python
from lumo import Meter,AvgMeter

am = AvgMeter() # use for record average
for j in range(500):
    meter = Meter()
    meter.percent(meter.c_) # when print, format 'c' as a percentage
    meter.a = 1
    meter.b = "2"
    meter.c = torch.rand(1)[0]

    meter.loss = loss_fn(...)
    meter.rand = torch.rand(2)
    meter.d = [4] # you can record any type of variable
    meter.e = {5: "6"}

    am.update(meter) # Update current value in meter. Average value will be calculated automatic by declaration and the type of the variable.
    print(am)
```


## Contribute

`lumo` will be better in the future, but there are still some lack exists currently, including:

 - **Lack of more detail guide** because of the lacking of developer's energy and time.
 - **Lack more tests**. unit test only covers a part of the code. I hope I fixed all bugs during my using of it, but there is no guarantee of it. The compatibility is also unguaranteed. So, welcome to [issus](https://github.com/sailist/lumo/issues) it if you find it.
 - **Lack of development experience**. So the version number may be confused.

Thanks for all contribution.



For file read/write and get/set, I designed

- [Params], to make runtime config get/set/load/dump easily,
- [globs], a global/local/runtime environment variables manager.
- [Saver], to help you save/load/manage your checkpoints/models in one class.

For data processing, I designed

- [Builder], to hold nearly all dataset formats and special operations by one class,

For managing experiments, I designed

- [Experiment], which can
    - make you build a suitable directory and file path in one place,
    - make you record lightweight data, and
    - help you make snapshot for your project code (based on git), which can make each result recoverable and
      reproducible
- [random manager], a cross-lib(random/numpy/pytorch) random seed manager

For log and meter variables produced during experiment, I designed

- [Meter] to meter every thing in appropriate format, and
- [Logger] to log every thing in appropriate format.

Finally, I designed [Trainer] to bundle all module above for deep learning experiment.


As you can see, These modules covered most demandings on deeplearning 

You can find what you want and click the link to quickly learn HOW TO USE it! All module is designed easy to use, it's
my principles.


