import os
from typing import List
import tarfile


def get_path_suffix(path1, path2):
    relpath = os.path.relpath(path1, path2)
    return os.path.splitext(relpath)[0]


def compress_dpath(paths: List[str], names: List[str], target, root_name=None):
    """
    Compress multiple directories into a single tar archive.

    Args:
        paths (List[str]): List of paths to the directories to be compressed.
        names (List[str]): List of names to be used for each directory in the tar archive.
        target (str): Path to the tar archive file to create.

    Returns:
        None

    Raises:
        IOError: If there is an error opening or writing to the target file.
        tarfile.TarError: If there is an error adding a file to the tar archive.

    Example:
        >>> compress_dpath(['A/dir', 'B/dir'], ['dir_a', 'dir_b'], 'archive.tar.gz')

    """
    assert target.endswith('tar')

    if isinstance(root_name, str):
        names = [os.path.join(root_name, i) for i in names]

    with tarfile.open(target, "w:gz") as tar:
        for path, name in zip(paths, names):

            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    tar.add(file_path, arcname=os.path.join(name, get_path_suffix(root, path), file))
    return target
