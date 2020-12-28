"""

"""


from thexp import globs
from torchvision import datasets
from thexp.contrib.data import SemiDataset

base = datasets.CIFAR10(globs["nddatasets"], train=True, download=False)



SemiDataset.create_indice(4000,5000,)