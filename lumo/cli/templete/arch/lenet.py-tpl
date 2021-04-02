"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self, out=10, with_fc=True):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.with_fc = with_fc
        self.fc3 = nn.Linear(84, out)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        if self.with_fc:
            out = self.fc3(out)
        return out

    def fc(self, mid) -> torch.Tensor:
        return self.fc3(mid)

    def softmax(self, logits):
        return F.softmax(logits, dim=1)


if __name__ == '__main__':
    import torch

    x = torch.rand(5, 1, 28, 28)
    print(LeNet()(x).shape)
