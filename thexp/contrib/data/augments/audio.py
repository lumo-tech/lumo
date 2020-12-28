import numpy as np
import torch


class RandomCrop():
    def __init__(self, pad=200):
        self.pad = pad

    def __call__(self, x):
        size = x.shape[0]
        x = np.pad(x, self.pad, mode='wrap')
        n = np.random.randint(0, self.pad)
        return x[n:n + size]


class RandomShift():
    def __call__(self, x):
        return np.roll(x, np.random.randint(0, x.shape[0] // 2), axis=-1)


class RandomNoisy():

    def __init__(self, rate=0.00001) -> None:
        super().__init__()
        self.rate = rate

    def __call__(self, x):
        wn = np.random.randn(*x.shape)
        return x + np.random.rand() * self.rate * wn


class RandomGain():
    def __init__(self, low=0.8, high=1.2):
        self.low = low
        self.high = high

    def __call__(self, x):
        return np.clip(x * (self.low + np.random.rand() * self.high), a_min=-32768, a_max=32767)


class PadTo():
    def __init__(self, to_size=20320):
        self.to_size = to_size

    def __call__(self, x):
        left = (self.to_size - x.shape[0]) // 2
        right = (self.to_size - x.shape[0]) - left
        return np.pad(x, (left, right), mode='constant', constant_values=0)


class ToTensor():
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float)
