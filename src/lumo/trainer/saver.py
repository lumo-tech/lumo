import os
import shutil
import textwrap
from typing import Optional, Union, List

from lumo.utils import safe_io as io


# state_dict_tuple = namedtuple('state_dict_tuplt', ['state_dict', 'meta_info'], defaults=[None])

class state_dict_tuple:
    """
    A class that stores a state dictionary and corresponding meta information.

    Args:
        state_dict (dict, optional): The state dictionary to be stored. Defaults to None.
        meta_info (Any, optional): Any meta information to be stored. Defaults to None.

    Attributes:
        state_dict (dict): The stored state dictionary.
        meta_info (Any): The stored meta information.

    Returns:
        A state_dict_tuple instance.

    Raises:
        IndexError: If the index is not 0 or 1.

    Examples:
    ```python
    # Create an instance of state_dict_tuple with state_dict and meta_info
    >>> sd = state_dict_tuple({'a': 1, 'b': 2}, 'meta')

    # Access the state_dict and meta_info using the [] operator
    >>> sd[0]
    {'a': 1, 'b': 2}
    >>> sd[1]
    'meta'
    ```
    """

    def __init__(self, state_dict=None, meta_info=None):
        self.state_dict = state_dict
        self.meta_info = meta_info

    def __getitem__(self, item):
        """
        Get the stored state dictionary or meta information.

        Args:
            item (int): Index of the desired item.

        Returns:
            The state dictionary (if item is 0) or the meta information (if item is 1).

        Raises:
            IndexError: If the index is not 0 or 1.

        Examples:
            # Access the state_dict and meta_info using the [] operator
            >>> sd = state_dict_tuple({'a': 1, 'b': 2}, 'meta')
            >>> sd[0]
            {'a': 1, 'b': 2}
            >>> sd[1]
            'meta'
            >>> sd[2]  # Raises IndexError
            IndexError: 2
        """
        if item == 0:
            return self.state_dict
        elif item == 1:
            return self.meta_info
        raise IndexError(item)


class Saver:
    """
    Write state_dict into test dirs, record save log into <repo working dir>/.lumo/save.<exp_name>.log

    format: <test_name> -> <fn>

    Raises:
        When save/load operations happend, you may meet Out Of Space, FileNotExist, or other problems,
        then Exception or its stack infomation will be printed in stderr (not be raised), and `False`/`None` value will be returned by
        save_xx and load_xx functions.
        You can handle these values and stop your program manully, or just let it go.
    """

    def __init__(self, save_dir):
        self._save_dir = save_dir

    @property
    def save_dir(self):
        os.makedirs(self._save_dir, exist_ok=True)
        return self._save_dir

    def _guess_abs_path(self, fn: str) -> str:
        path = fn
        if os.path.basename(fn) == fn:
            path = os.path.join(self.save_dir, fn)
        return path

    def _create_state_dict_name(self, step=0, replacement: bool = True,
                                prefix=None, ignore_number=False, suffix='pt') -> str:
        """
        create path for saving state_dict
        """
        offset = 0

        def _create_path():
            res = []
            res.append(prefix)
            if not replacement:
                res.append(f"{offset:01d}")
            if ignore_number:
                res.append(None)
            else:
                res.append(f"{step:06d}")
            res.append(suffix)
            fn = '.'.join(list(filter(lambda x: x is not None, res)))
            return os.path.join(self.save_dir, fn)

        path = _create_path()
        if not replacement:
            while os.path.exists(path):
                offset += 1
                path = _create_path()
        return path

    def dump_state_dict(self, obj, fn, meta_info: Union[str, dict] = None):
        """

        Args:
            obj: picklable object
            fn: path-like object
            meta_info: meta infomation for the dumped object.

        Returns:
            saved filepath, None if something went wrong.
        """
        io.dump_state_dict(obj, fn)
        if meta_info is not None:
            if isinstance(meta_info, str):
                meta_info = {'msg': meta_info}
            json_fn = f"{fn}.json"
            io.dump_json(meta_info, json_fn)
        return fn

    def load_state_dict(self, fn: str, with_meta=False, map_location='cpu') -> state_dict_tuple:
        """

        Args:
            fn: pickled file path.
            with_meta: whether to get its meta infomation at the same time, default is False.
            map_location:

        Returns:
            If `with_meta=True`, a `state_dict_tuple` object will be returned, you can unpack this tuple directly:
            ```
            res = load_state_dict(...)
            ckpt, info = res
            ```
            or `ckpt` object will be returned alone.

            if something went wrong, `ckpt` or `info` will be a None object.
        """
        path = self._guess_abs_path(fn)
        ckpt = io.load_state_dict(path, map_location=map_location)

        if not with_meta:
            return ckpt

        info = self.load_meta_info(fn)
        return state_dict_tuple(ckpt, info)

    def load_meta_info(self, fn: str) -> dict:
        """
        load
        Args:
            fn: pickled file path, not the metainfo file path(which ends with suffix '.json').

        Returns:
            info object, or None if something went wrong.

        """
        path = self._guess_abs_path(fn)
        info_path = f"{path}.json"
        info = io.load_json(info_path)
        return info

    def save_keypoint(self, steps: int, state_dict, meta_info: Union[str, dict] = None) -> str:
        """
        save a object which is keypoint.
        Args:
            steps:
            state_dict:
            meta_info:

        Returns:

        """
        path = self._create_state_dict_name(steps, replacement=True, prefix='key')
        self.dump_state_dict(state_dict, path, meta_info=meta_info)
        return path

    def save_checkpoint(self, step: int, state_dict, meta_info: Union[str, dict] = None,
                        max_keep=10, is_best=False) -> Optional[str]:
        """
        save a object as a deleteable

        Args:
            step: progress indecator, using global_steps or epoch count is recommanded.
            state_dict: state dict
            meta_info: meta information for this state dict
            max_keep:
            is_best: Whether this checkpoint is currently best. If True, the best checkpoint
                file will be replaced by this one.

        Returns:
            Saved checpoint path, or None if something is wrong.

        """
        path = self._create_state_dict_name(step, replacement=False, prefix='checkpoints')
        res = self.dump_state_dict(state_dict, path, meta_info=meta_info)
        del state_dict
        history = self.list_checkpoints()
        if len(history) > max_keep:
            [os.remove(os.path.join(self.save_dir, i)) for i in history[:-max_keep]]

        if res:
            if is_best:
                best_path = self._create_state_dict_name(step,
                                                         replacement=True,
                                                         prefix='best.checkpoint',
                                                         ignore_number=True)
                shutil.copy2(path, best_path)
                src_json_fn = f"{path}.json"
                if os.path.exists(src_json_fn):
                    tgt_json_fn = f"{best_path}.json"
                    shutil.copy2(src_json_fn, tgt_json_fn)

            return path
        else:
            return None

    def save_model(self, step: int, state_dict, meta_info: Union[str, dict] = None, is_best=False) -> str:
        """
        save model

        Args:
            steps: progress indecator, using global_steps or epoch count is recommanded.
            state_dict: state dict
            meta_info: meta information for this state dict
            is_best: Whether this checkpoint is currently best. If True, the best checkpoint
                file will be replaced by this one.

        Returns:
            Saved checpoint path, or None if something is wrong.

        """
        path = self._create_state_dict_name(step, replacement=True, prefix='model')
        res = self.dump_state_dict(state_dict, path, meta_info=meta_info)
        del state_dict
        if res:
            if is_best:
                best_path = self._create_state_dict_name(step,
                                                         replacement=True,
                                                         prefix='best.model',
                                                         ignore_number=True)
                shutil.copy2(path, best_path)
                src_json_fn = f"{path}.json"
                if os.path.exists(src_json_fn):
                    tgt_json_fn = f"{best_path}.json"
                    shutil.copy2(src_json_fn, tgt_json_fn)
            return path
        else:
            return None

    def load_checkpoint(self, index=-1, best_if_exist=False, fn=None, with_meta=False, map_location='cpu'):
        """
        Loads a checkpoint file.

        Args:
            index (int, optional): Index of the checkpoint file in the list of checkpoints. Defaults to -1.
            best_if_exist (bool, optional): If True, load the best checkpoint file. Defaults to False.
            fn (str, optional): The filename of the checkpoint file. Defaults to None.
            with_meta (bool, optional): If True, the checkpoint file is expected to contain metadata. Defaults to False.
            map_location (str, optional): Where to load the checkpoint file. Defaults to 'cpu'.

        Returns:
            Union[None, Any]: None if the checkpoint file could not be loaded, otherwise the loaded checkpoint.

        """
        if fn is None and best_if_exist:
            fn = self.best_checkpoint()
        if fn is None:
            try:
                fn = self.list_checkpoints()[index]
            except:
                pass
        if fn is not None:
            return self.load_state_dict(fn, with_meta, map_location)
        return None

    def load_keypoint(self, index=-1, fn=None, with_meta=False, map_location='cpu'):
        """
        Loads a checkpoint file that is key.

        Args:
            index (int, optional): Index of the keypoint file in the list of keypoints. Defaults to -1.
            fn (str, optional): The filename of the keypoint file. Defaults to None.
            with_meta (bool, optional): If True, the keypoint file is expected to contain metadata. Defaults to False.
            map_location (str, optional): Where to load the keypoint file. Defaults to 'cpu'.

        Returns:
            Union[None, Any]: None if the keypoint file could not be loaded, otherwise the loaded keypoint.

        """
        if fn is None:
            try:
                fn = self.list_keypoints()[index]
            except:
                pass
        if fn is not None:
            return self.load_state_dict(fn, with_meta, map_location)
        return None

    def load_model(self, index=-1, best_if_exist=False, fn=None, with_meta=False, map_location='cpu'):
        """
        Loads a model file.

        Args:
            index (int, optional): Index of the model file in the list of models. Defaults to -1.
            best_if_exist (bool, optional): If True, load the best model file. Defaults to False.
            fn (str, optional): The filename of the model file. Defaults to None.
            with_meta (bool, optional): If True, the model file is expected to contain metadata. Defaults to False.
            map_location (str, optional): Where to load the model file. Defaults to 'cpu'.

        Returns:
            Union[None, Any]: None if the model file could not be loaded, otherwise the loaded model.

        """
        if fn is None and best_if_exist:
            fn = self.best_model()
        if fn is None:
            try:
                fn = self.list_models()[index]
            except:
                pass
        if fn is not None:
            return self.load_state_dict(fn, with_meta, map_location)
        return None

    def best_checkpoint(self):
        """
        Returns the filename of the best checkpoint file if it exists, otherwise None.

        Returns:
            Union[None, str]: The filename of the best checkpoint file if it exists, otherwise None.

        """
        fn = os.path.join(self.save_dir, 'best.checkpoint.pt')
        if os.path.exists(fn):
            return fn
        return None

    def best_model(self):
        """Returns the filename of the best model file if it exists, otherwise None."""
        fn = os.path.join(self.save_dir, 'best.model.pt')
        if os.path.exists(fn):
            return fn
        return None

    def _is_pkl(self, x: str, start, end):
        """Helper function that returns True if the string x starts with start and ends with end."""
        return x.startswith(start) and x.endswith(end)

    def list_checkpoints(self) -> List[str]:
        """Returns a sorted list of filenames of all checkpoint files in the save directory."""
        return sorted(list(filter(lambda x: self._is_pkl(x, 'checkpoints', 'pt'),
                                  os.listdir(self.save_dir))),
                      key=lambda x: os.stat(os.path.join(self.save_dir, x)).st_ctime)

    def list_keypoints(self) -> List[str]:
        """Returns a sorted list of filenames of all keypoint files in the save directory."""
        return sorted(list(filter(lambda x: self._is_pkl(x, 'key', 'pt'),
                                  os.listdir(self.save_dir))),
                      key=lambda x: os.stat(os.path.join(self.save_dir, x)).st_ctime)

    def list_models(self) -> List[str]:
        """Returns a sorted list of filenames of all model files in the save directory."""
        return sorted(list(filter(lambda x: self._is_pkl(x, 'model', 'pt'),
                                  os.listdir(self.save_dir))),
                      key=lambda x: os.stat(os.path.join(self.save_dir, x)).st_ctime)

    def list(self):
        """Returns a sorted list of all filenames in the save directory."""
        return sorted(os.listdir(self.save_dir))

    def summary(self):
        """Prints a summary of the contents of the save directory."""
        print(f"saved in : {self.save_dir}")
        print(textwrap.indent('\n'.join(self.list()), ' - '))
