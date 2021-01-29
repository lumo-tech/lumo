from typing import Dict, Callable, Tuple

from PIL import Image
from thexp import globs
from thexp.decorators import regist_func
from thexp.base_classes import llist
from torchvision.datasets.cifar import CIFAR10
import numpy as np

# globs.add_value('datasets', 'path/to/all_datasets/', level=globs.LEVEL.globals)
root = globs['datasets']

datasets = {
    # 'cifar10': cifar10,
}  # type:Dict[str,Callable[[bool],Tuple[llist,llist]]]


@regist_func(datasets)
def cifar10(train=True):
    dataset = CIFAR10(root=root, train=train)
    xs = llist(Image.fromarray(i) for i in dataset.data)
    ys = np.array(int(i) for i in dataset.targets)

    return xs, ys
