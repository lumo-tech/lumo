"""
runtime environment constant.

Pycharm debug mode currently has a bug that can't run next step after a breakpoint when have multithread(num_workers>0).

So the parameter can be conditional set as:

```python
from thexp import ENV
num_workers = 0 is ENV.IS_PYCHARM_DEBUG else 4
```

The class ENVIRON_ is also embeded as a class variable named `ENV` in class Params, means you can use like:

```python
from thexp import Params
Params.ENV.IS_PYCHARM_DEBUG
```
"""
import os
from sys import platform


class ENVIRON_:
    IS_WIN = (platform == "win32")
    IS_MAC = (platform == "darwin")
    IS_LINUX = (platform == "linux" or platform == "linux2")

    IS_REMOTE = any([i in os.environ for i in ['SHELL',
                                               'SHLVL',
                                               'SSH_CLIENT',
                                               'SSH_CONNECTION',
                                               'SSH_TTY']])
    IS_LOCAL = not IS_REMOTE

    IS_PYCHARM = os.environ.get("PYCHARM_HOSTED", 0) == "1"
    IS_PYCHARM_DEBUG = eval(os.environ.get('IPYTHONENABLE', "False"))


