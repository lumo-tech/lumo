import copy
import warnings
from functools import partial
from pprint import pformat
from copy import copy as builtin_copy
from typing import Callable, NewType, Dict, Any, Iterable, Sequence

import numpy as np
from torch.utils.data import Dataset

from lumo.data.loader import LumoDataLoader

SingleValueTransform = NewType('SingleValueTransform', Callable[[Any], Any])
DictTransform = NewType('DictTransform', Callable[[Dict[str, Any]], Any])


class DatasetBuilder(Dataset):
    """
    A subclass of the Dataset class designed for quick and common dataset construction.

    For instance, a simple CIFAR-10 Dataset in (x, y) format can be easily created using the following code:
    ```python
    from torchvision.datasets.cifar import CIFAR10
    ds = CIFAR10(...)
    x,y = ds[0]
    ```
    However, there may be requirements to modify the format, such as (x1, x2, y), due to tricks like cutmix or multiple argument transformers. In this case, we need to extend the CIFAR10 class as follows:
    ```python
    class MyCIFAR10(CIFAR10):
        def __getitem__(self,index):
            ...
            return x1, x2, y
    ```

    If we have multiple format requirements, we can either add extra arguments to the MyCIFAR10 class or create more subclasses, but both methods are redundant.

    Now, by using the DatasetBuilder, this process can be simplified to the utmost.

    To begin, we need to prepare the data source/inputs for CIFAR-10, which includes images and targets. We can register these inputs using .add_input(source, name) as follows:

    ```python
    source = CIFAR10()
    ds = (
        DatasetBuilder()
            .add_input('xs', source.data)
            .add_input('ys', source.targets)
        )
    ```

    Next, we define the outputs. If we want the output format to be (xs, ys), we can use the following code:
    ```python
    (ds
        .add_output('xs','xs')
        .add_output('ys','ys')
    )
    ```
    The function `.add_output(source_name, output_name)` defines a data flow from input to output.
    In `.add_output('xs', 'xs')`, the input is `source.data` named as 'xs', and the output is also named as 'xs'.
    If we want to rename the output name to 'xs1', we can use .add_output('xs', 'xs1').

    Now you can see the benefits of this approach. If you need an output format like `(xs1, xs2, ys)`, you just need to modify the code as follows:
    ```python
    (ds
        .add_output('xs','xs1')
        .add_output('xs','xs2')
        .add_output('ys','ys')
    )
    ```

    Besides, you can access the index of each data by `.add_idx('name')`.
    ```
    ds.add_idx('idx')
    print(ds[0])
    {'idx': 0, ...others...}
    ```

    Finally, we can use transforms. Each input and output can be passed a transform parameter during definition, such as:
    ```python
    ds.add_input('xs', xs, transform=default_loader)
    ds.add_output('xs','xs1',transform=randargument)
    ds.add_output('xs','xs2',transform=weak)
    ```

    The transform defined at the input stage will only be called once when there is a corresponding output. Each output has its own transform.
    That is to say, the transform execution process defined by the above code will like:
    ```python
    x -> default_loader -> randargument -> xs1
                       \-> weak -> xs2
    ```

    It's possible that you may be confused about the usage of output names. Commonly, the output types of `dataset[index]` are `list` or `dict`.
    DatasetBuilder provides both types for everyone, and by default, the dict type is used. When defining outputs as `(xs1, xs2, ys)`, the output of `ds[index]` should be:
    ```python
    {'xs1': np.array, 'xs2': np.array, 'ys': 1}
    ```

    You can change the output type to list by calling `.chain()`:
    ...python
    ds.chain()
    xs1,xs2,ys = ds[index]
    ```



    """

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
        builder = DatasetBuilder()
        builder._prop = builtin_copy(self._prop)
        builder._idx_keys = builtin_copy(self._idx_keys)
        builder._data = builtin_copy(self._data)
        builder._outs = builtin_copy(self._outs)
        builder._transforms = builtin_copy(self._transforms)
        builder._outkeys = builtin_copy(self._outkeys)
        builder._iter_cache = builtin_copy(self._iter_cache)
        return builder

    def subset(self, indices: Sequence[int], copy=False):
        if copy:
            builder = self.copy()
        else:
            builder = self
        builder._prop['subindices'] = np.array(indices)
        builder._update_len()
        return builder

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
            raise ValueError('subset() should be called after preparing all data source.')
        if self.pseudo_repeat is not None or self.pseudo_length is not None:
            raise ValueError('scale_to_size() or repeat() should be called after preparing all data source.')

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
                    raise ValueError(f'New added source `{name}` has different length from '
                                     f'other registed source.(size {size} != newsize {newsize})')

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
        """
        Register a input source with the transform (if provided).
        Args:
            name: source name
            source: source, should be a sized object.
            transform:


        Notes:
            Iterable object without `__len__` method currently are not well-tested. Be careful to use them in DatasetBuilder.

        """
        assert name not in self._data, f'Source name {name} duplicated.'
        self._check_source(name, source)
        self._data[name] = source
        self._transforms[name] = transform
        return self

    def add_input_transform(self, name: str, transform: SingleValueTransform = None):
        assert name in self._data, f'Source {name} should be added.'
        warnings.warn('`add` may cause confusion, use set_input_transform ')
        return self.set_input_transform(name, transform)

    def add_output(self, name: str, outkey: str, transform: SingleValueTransform = None):
        """
        Add a data flow from inputs[name] to outputs[outkey] with the transform (if provided).
        Args:
            name: source name of inputs
            outkey: output name of output
            transform: a callable function

        Returns:

        """
        assert name in self._data, f'Must have data source {name} first.'

        outkeys = self._outs.setdefault(name, list())
        assert outkey not in self._outkeys, f'Output key {outkey} duplicated.'
        outkeys.append(outkey)
        self._outkeys.append(outkey)

        self._transforms[f'::{outkey}'] = transform
        return self

    def add_output_transform(self, outkey: str, transform: SingleValueTransform = None):
        """
        Add or **replace** transform of the output name.
        Args:
            outkey: output name.
            transform: a callable function
        """
        assert outkey in self._outkeys, f'Output key {outkey} should be added.'
        warnings.warn('add may cause confusion, use set_output_transform ')
        return self.set_output_transform(outkey, transform)

    def add_global_transform(self, transform: DictTransform):
        self._transforms['::global::'] = transform
        return self

    def set_input_transform(self, name, transform: SingleValueTransform = None):
        """
        Add or **replace** transform of the input source {name}.
        Args:
            name: source name.
            transform: a callable function

        """
        self._transforms[name] = transform
        return self

    def set_output_transform(self, outkey, transform: SingleValueTransform = None):
        """
        Add or **replace** transform of the output {name}.
        Args:
            outkey: output name.
            transform: a callable function

        """
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
