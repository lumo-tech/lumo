"""

"""
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.fakedata import FakeData
from torchvision.transforms import ToTensor

from thexp import DataBundler

bundler = DataBundler()

sub = DataBundler()
sub.add(DataLoader(FakeData(transform=ToTensor()), batch_size=10)) \
    .add(DataLoader(FakeData(image_size=(3, 32, 32), transform=ToTensor()), batch_size=10)) \
    .zip_mode()

bundler.add(sub) \
    .add(DataLoader(FakeData(image_size=(3, 28, 28), transform=ToTensor()), batch_size=10)) \
    .zip_mode()

for ((i1, l1), (i2, l2)), (i3, l3) in bundler:
    print(i1.shape, l1.shape, i2.shape, l2.shape, i3.shape, l3.shape)

bundler = (
    DataBundler()
        .cycle(DataLoader(FakeData(size=10, image_size=(3, 28, 28), transform=ToTensor()), batch_size=10))
        .add(DataLoader(FakeData(size=1000, image_size=(3, 28, 28), transform=ToTensor()), batch_size=10))
        .zip_mode()
)


