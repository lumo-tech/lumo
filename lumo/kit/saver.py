import os
import shutil
import textwrap
from collections import namedtuple
from typing import Optional

from lumo.utils import safe_io as io

state_dict_tuple = namedtuple('state_dict_tuplt', ['state_dict', 'meta_info'], defaults=[None])


class Saver:
    """
    Write state_dict into test dirs, record save log into <repo working dir>/.lumo/save.<exp_name>.log

    format:
        <test_name> -> <fn>

    Raises:
        When save/load operations happend, you may meet Out Of Space, FileNotExist, or other problems,
        then Exception or its stack infomation will be printed in stderr (not be raised), and `False`/`None` value will be returned by
        save_xx and load_xx functions.
        You can handle these values and stop your program manully, or just let it go.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

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
            res.append(f"{offset:01d}")
            if ignore_number:
                res.append(None)
            else:
                res.append(f"{step:06d}")
            res.append(suffix)
            return '.'.join(res)

        path = _create_path()
        if not replacement:
            while os.path.exists(path):
                offset += 1
                path = _create_path()
        return path

    def dump_state_dict(self, obj, fn, meta_info: dict = None):
        """

        Args:
            obj: picklable object
            fn: path-like object
            meta_info: meta infomation for the dumped object.

        Returns:
            saved filepath, None if something went wrong.
        """
        fn = io.dump_state_dict(obj, fn)
        if meta_info is not None:
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
        ckpt = io.load_state_dict(path)

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

    def save_keypoints(self, steps: int, state_dict, meta_info: dict = None) -> str:
        """
        save a object which is keypoint.
        Args:
            steps:
            state_dict:
            meta_info:

        Returns:

        """
        path = self._create_state_dict_name(steps, replacement=False, prefix='key')
        self.dump_state_dict(state_dict, path, meta_info=meta_info)
        return path

    def save_checkpoints(self, step: int, state_dict, meta_info: dict = None,
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
        path = self._create_state_dict_name(step, replacement=True, prefix='checkpoints')
        res = self.dump_state_dict(state_dict, path, meta_info=meta_info)
        if res:
            if is_best:
                best_path = self._create_state_dict_name(step,
                                                         replacement=True,
                                                         prefix='best.checkpoints',
                                                         ignore_number=True)
                shutil.copy2(path, best_path)
            return path
        else:
            return None

    def save_model(self, step: int, state_dict, meta_info: dict = None, is_best=False) -> str:
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
        if res:
            if is_best:
                best_path = self._create_state_dict_name(step,
                                                         replacement=True,
                                                         prefix='best.model',
                                                         ignore_number=True)
                shutil.copy2(path, best_path)
            return path
        else:
            return None

    def list(self):
        return sorted(os.listdir(self.save_dir))

    def summary(self):
        print(f"saved in : {self.save_dir}")
        print(textwrap.indent('\n'.join(self.list()), ' - '))
