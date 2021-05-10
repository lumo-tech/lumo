from typing import Sequence, Union, Dict, Any
import os
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset


class Data:
    def __init__(self, value, name=None):
        self.name = name
        self.value = value


class DelegateDataTypeError(RuntimeError): pass


class DataDelegate(Dataset):
    """
    """

    def __init__(self, source, len=None, len_func=None):
        self.source = source
        self.len = len
        self.len_func = len_func

    def __len__(self):
        if self.len_func is not None:
            return self.len_func(self.source)
        if self.len is not None:
            return self.len
        return len(self.source)

    def __getitem__(self, item) -> Union[Data, Sequence[Data], Dict[str, Data], Dict[str, Any]]:
        raise NotImplementedError()

    def create_data(self, value, name=None) -> Data:
        return Data(value=value, name=name)


class LambdaDelegate(DataDelegate):
    """
    It's not recommaned to use lambda function to feed `get_func` or `len_func`, it will raise an error
    in MultiProcess mode cause <lambda> function is unpicklable object.
    """

    def __init__(self, name, get_func, source, len=None, len_func=None, wrap=True):
        super().__init__(source, len, len_func)
        self.name = name
        self.get_func = get_func
        self.wrap = wrap

    def __getitem__(self, index):
        val = self.get_func(index)
        if self.wrap:
            return Data(val, self.name)
        return val


class ClassSampleDelegate(DataDelegate):

    def __init__(self, source, len=None, len_func=None):
        super().__init__(source, len, len_func)
        self._n_classes = list()
        self._neg_classes = dict()
        self._cls_dict = {}

    def _create_cls_dict(self, cls_squence: Sequence[int]):
        if isinstance(cls_squence, torch.Tensor):
            cls_squence = cls_squence.reshape(-1).detach().cpu().numpy()
        elif isinstance(cls_squence, np.ndarray):
            cls_squence = cls_squence.view(-1)
        elif isinstance(cls_squence, list):
            cls_squence = np.array(cls_squence)
        else:
            assert False

        self._n_classes = list(set(cls_squence))
        neg_cls = {i: list(self._n_classes) for i in self._n_classes}
        _ = {neg_cls[i].remove(i) for i in neg_cls}
        self._neg_classes = neg_cls

        for i in self._n_classes:
            self._cls_dict[i] = np.where(cls_squence == i)[0]

    def _rand_choice_same_cls(self, cls):
        return np.random.choice(self._cls_dict[cls], 1)

    def _rand_choice_neg_cls(self, cls):
        neg_cls = np.random.choice(self._neg_classes[cls], 1)
        return np.random.choice(self._cls_dict[neg_cls], 1)


class BaseHDF5Builder(DataDelegate):
    def __init__(self, h5path: str):
        super().__init__(None)
        import h5py
        self.h5path = h5path
        try:
            self.h5file = h5py.File(h5path, 'r')
        except OSError:
            self._create_tmp_h5(h5path)
        self.initial_h5attr()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def _create_tmp_h5(self, h5path):
        import h5py
        c = 0
        h5file = None
        while h5file is not None:
            nh5path = f"{h5path}_builder_tmp{c}.h5"
            try:
                if not os.path.exists(nh5path):
                    shutil.copy(h5path, nh5path)
                h5file = h5py.File(nh5path, 'r')
            except OSError:
                c += 1
        return h5file

    def initial_h5attr(self):
        pass
