import random
from lumo.base_classes.list import llist


class MemoryBank:
    def __init__(self, queue_size=512, lis_cls=llist, *args, **kwargs):
        self.queue_size = queue_size
        assert self.queue_size > 0, 'queue size must be larger than 0.'
        self._lis = lis_cls()

    def __len__(self):
        return len(self._lis)

    def __iter__(self):
        return iter(self._lis)

    def __getitem__(self, item):
        return self._lis[item]

    def isempty(self):
        return len(self._lis) == 0

    def isfull(self):
        return len(self._lis) == self.queue_size

    def pop(self, index=0, default=None):
        if self.isempty() or index >= len(self):
            return default
        return self._lis.pop(index)

    def top(self):
        return self._lis[0]

    def topk(self, k=1):
        return self._lis[:k]

    def tail(self):
        return self._lis[-1]

    def tailk(self, k=1):
        return self._lis[-k:]

    def push(self, item):
        self._lis.append(item)
        if self.isfull():
            self.pop()

    def choice(self, default=None):
        if self.isempty():
            return default
        return random.choice(self._lis)

    def apply(self, func):
        self._lis = self._lis.__class__(func(i) for i in self._lis)
