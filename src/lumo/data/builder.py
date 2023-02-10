import copy
import warnings
from functools import partial
from pprint import pformat

from typing import Callable, NewType, Dict, Any, Iterable, Sequence

import numpy as np
from torch.utils.data import Dataset

from lumo.data.loader import LumoDataLoader

SingleValueTransform = NewType('SingleValueTransform', Callable[[Any], Any])
DictTransform = NewType('DictTransform', Callable[[Dict[str, Any]], Any])


class DatasetBuilder(Dataset):
    def __init__(self):
        self._prop = {}

        self._idx_keys = set()
        self._data = {}
        self._outs = {}
        self._transforms = {}

        self._outkeys = []

        self._iter_cache = {}

    def __repr__(self):

        if self.sized:
            return f'Builder(flow={pformat(self._outs)}, sized={self.sized}, size={len(self)}, iterable={self.iterable})'
        else:
            return f'Builder(flow={pformat(self._outs)}, sized={self.sized}, iterable={self.iterable})'

    def __iter__(self):
        if self.subindices is not None:
            warnings.warn('subset() has no effects in iter mode.')

        for key, outkeys in self._outs.items():
            if key == '::idx::':
                warnings.warn(f'iter does not support idx, will skip {outkeys}')
                continue

            source = self._data[key]
            for outkey in outkeys:
                self._iter_cache[outkey] = iter(source)
        return self

    def __next__(self):
        if len(self._iter_cache) == 0:
            raise StopIteration()
        try:
            outputs = {k: next(v) for k, v in self._iter_cache.items()}
            if self.mode != 'zip':
                outputs = [outputs[outkey] for outkey in self._outkeys]
            return outputs
        except StopIteration as e:
            self._iter_cache.clear()
            raise e

    def __getitem__(self, index):

        index = self.map_index(index)

        outputs = {}

        for key, outkeys in self._outs.items():
            if key == '::idx::':
                ipt = index
            else:
                ipt = self._data[key][index]
            ipt_transform = self._transforms.get(key, None)
            if ipt_transform is not None:
                ipt = ipt_transform(ipt)

            for outkey in outkeys:
                opt = ipt
                opt_transform = self._transforms.get(f'::{outkey}', None)
                if opt_transform is not None:
                    opt = opt_transform(opt)

                outputs[outkey] = opt

        glb_transform = self._transforms.get('::global::', None)
        if glb_transform is not None:
            outputs = glb_transform(outputs)

        if self.mode == 'chain':
            outputs = [outputs[outkey] for outkey in self._outkeys]
        elif self.mode == 'item':
            outputs = list(outputs.values())[0]
        return outputs

    def __len__(self):
        if not self.sized:
            raise TypeError('DatasetBuilder can not be sized when data source is empty or only iterable.')

        return self._prop['__clen__']

    def _update_len(self):
        if not self.sized:
            self._prop['__clen__'] = None
            return

        if self.pseudo_length is not None:
            res = self.pseudo_length
            self._prop['__clen__'] = res
            return res

        if self.subindices is not None:
            res = len(self.subindices)
        else:
            res = self._prop['__len__']

        if self.pseudo_repeat is not None:
            res = res * self.pseudo_repeat
        self._prop['__clen__'] = res
        return res

    @property
    def inputs(self):
        return self._data

    @property
    def outputs(self):
        mapping = {}
        for key, outkeys in self._outs.items():
            if key == '::idx::':
                source = range(len(self))
            else:
                source = self._data[key]

            for outkey in outkeys:
                mapping[outkey] = source
        return mapping

    @property
    def mode(self):
        return self._prop.get('mode', 'zip')

    @property
    def iterable(self):
        return self._prop.get('iterable', False)

    @property
    def subindices(self):
        return self._prop.get('subindices', None)

    @property
    def pseudo_length(self) -> int:
        return self._prop.get('pseudo_length', None)

    @property
    def pseudo_repeat(self) -> int:
        return self._prop.get('pseudo_repeat', None)

    def copy(self):
        db = DatasetBuilder()
        db._prop = copy.copy(self._prop)
        db._idx_keys = copy.copy(self._idx_keys)
        db._data = copy.copy(self._data)
        db._outs = copy.deepcopy(self._outs)
        db._transforms = copy.copy(self._transforms)
        db._outkeys = copy.copy(self._outkeys)
        return db

    def subset(self, indices: Sequence[int]):
        self._prop['subindices'] = np.array(indices)
        self._update_len()
        return self

    def scale_to_size(self, size: int):
        assert isinstance(size, int)
        assert 'pseudo_repeat' not in self._prop
        assert 'pseudo_length' not in self._prop

        self._prop['pseudo_length'] = size
        self._update_len()
        return self

    def repeat(self, multiple: int):
        assert isinstance(multiple, int)
        assert 'pseudo_length' not in self._prop
        assert 'pseudo_repeat' not in self._prop

        self._prop['pseudo_repeat'] = multiple
        self._update_len()
        return self

    def map_index(self, index):
        """
        Map the raw index to the final index for source data.
        Args:
            index:

        Returns:

        """
        if self.pseudo_length is not None or self.pseudo_repeat is not None:
            if self.subindices is not None:
                index = index % len(self.subindices)
                index = self.subindices[index]
            else:
                index = index % self._prop['__len__']
        elif self.subindices is not None:
            index = self.subindices[index]

        return index

    @property
    def sized(self):
        return self._prop.get('sized', False)

    def chain(self):
        self._prop['mode'] = 'chain'
        return self

    def item(self):
        self._prop['mode'] = 'item'
        return self

    def zip(self):
        self._prop['mode'] = 'zip'
        return self

    def _check_source(self, name, source):
        # source is sized can be itered
        # source can be itered not meant it is sizable.
        if self.subindices is not None:
            return ValueError('subset() should be called after preparing all data source.')
        if self.pseudo_repeat is not None or self.pseudo_length is not None:
            return ValueError('scale_to_size() or repeat() should be called after preparing all data source.')

        if self._prop.get('sized', True):
            size = self._prop.get('__len__', None)
            try:
                newsize = len(source)
                if size is None:
                    self._prop['sized'] = True
                    self._prop['iterable'] = True
                    self._prop['__len__'] = newsize
                    self._update_len()
                elif size != newsize:
                    warnings.warn(f'Source {name} has different length from '
                                  f'other registed source.')

            except (TypeError, NotImplementedError):
                self._prop['sized'] = False
                self._prop['__len__'] = None

        if isinstance(source, Iterable):
            self._prop['iterable'] = True

        if not self._prop.get('iterable', False) and not self._prop.get('sized', False):
            raise TypeError(f'Source {name} must be an iterable or sized object, but got {type(source)}.')

    def add_idx(self, name):
        outkeys = self._outs.setdefault(f"::idx::", [])
        assert name not in self._outkeys, f'Output key {name} duplicated.'
        outkeys.append(name)
        self._outkeys.append(name)
        return self

    def add_input(self, name: str, source, transform: SingleValueTransform = None):
        assert name not in self._data, f'Source name {name} duplicated.'
        self._check_source(name, source)
        self._data[name] = source
        self._transforms[name] = transform
        return self

    def add_input_transform(self, name: str, transform: SingleValueTransform = None):
        assert name in self._data, f'Source {name} should be added.'
        self._transforms[name] = transform
        return self

    def add_output(self, name: str, outkey: str, transform: SingleValueTransform = None):
        assert name in self._data, f'Must have data source {name} first.'

        outkeys = self._outs.setdefault(name, list())
        assert outkey not in self._outkeys, f'Output key {outkey} duplicated.'
        outkeys.append(outkey)
        self._outkeys.append(outkey)

        self._transforms[f'::{outkey}'] = transform
        return self

    def add_output_transform(self, outkey: str, transform: SingleValueTransform = None):
        assert outkey in self._outkeys, f'Output key {outkey} should be added.'
        self._transforms[f'::{outkey}'] = transform
        return self

    def add_global_transform(self, transform: DictTransform):
        self._transforms['::global::'] = transform
        return self

    def set_input_transform(self, name, transform: SingleValueTransform):
        self._transforms[name] = transform
        return self

    def set_output_transform(self, outkey, transform: SingleValueTransform):
        self._transforms[f'::{outkey}'] = transform
        return self

    DataLoader = LumoDataLoader
    DL = LumoDataLoader  # alias to DataLoader

    def __getattribute__(self, item):
        res = super(DatasetBuilder, self).__getattribute__(item)
        if item == 'DataLoader':
            res = partial(res, dataset=self)
        return res

    def get_source(self, name):
        return self._data[name]
