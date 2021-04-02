"""

"""

import sys
sys.path.insert(0,"../../")
import time

from lumo.utils.screen import ScreenStr
s = "\rLong Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text"
print(ScreenStr(s,leftoffset=10),end="")
for i in range(100):
    time.sleep(0.2)


from lumo.contrib.data.collate import AutoCollate
from torch.utils.data.dataloader import DataLoader
import torch
device = torch.device('cuda:1')
DataLoader(...,collate_fn=AutoCollate(device))
