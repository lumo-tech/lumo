from typing import List
import numpy as np
import torch
from torch._utils import _accumulate
from torch.utils.data import Dataset


def sequence_split(lengths: List[int]) -> List[Dataset]:
    """
    对应于 torch.utils.data.dataset.random_split ，用于按照长度顺序切分数据集
    Args:
        lengths:

    Returns:

    """
    indices = torch.arange(0, sum(lengths)).tolist()
    return [indices[offset - length:offset] for offset, length in
            zip(_accumulate(lengths), lengths)]


def semi_split(labels, n_percls, val_size=10000, include_sup=True, repeat_sup=True, shuffle=True):
    """
    在半监督的情况下切分训练集，有标签训练集和无标签训练集
    """
    labels = np.array(labels)
    n_cls = len({int(i) for i in labels})

    n_per_un = (len(labels) - val_size) // n_cls
    indexs = []
    un_indexs = []
    val_indexs = []
    for i in range(n_cls):
        idx = np.where(labels == i)[0]

        np.random.shuffle(idx)
        indexs.extend(idx[:n_percls])

        if include_sup:
            un_indexs.extend(idx[:n_per_un])  # 无标签样本部份也使用有标签样板
        else:
            un_indexs.extend(idx[n_percls:n_per_un])

        val_indexs.extend(idx[n_per_un:])

    if repeat_sup:
        indexs = np.hstack([indexs for i in range((len(un_indexs) // len(indexs)) + 1)])
        indexs = indexs[:len(un_indexs)]

    if shuffle:
        np.random.shuffle(indexs)
        np.random.shuffle(un_indexs)
    return indexs, un_indexs, val_indexs


def train_val_split(target, val_size=10000, train_size=None):
    import numpy as np
    size = len(target)
    idx = np.arange(size)
    if isinstance(target, list):
        from lumo.base_classes import llist
        target = llist(target)

    idx = np.arange(len(target))
    np.random.shuffle(idx)

    if train_size is not None:
        assert size > val_size + train_size, "should less than {}, but {}".format(size, train_size + val_size)
        return idx[val_size:val_size + train_size], idx[:val_size]

    return idx[val_size:], idx[:val_size]


def ratio2length(total_len, *ratios) -> List[int]:
    """
    将比率转换成具体整数值
    Args:
        total_len:  总长度
        *ratios:  任意长度比率，和不要求为1

    Returns:
        List[int]

    """
    return [int(total_len * i) for i in ratios]


def create_imblanced_data(labels, imb_type='exp', imb_factor=0.02):
    """
    reimp code from https://github.com/dvlab-research/Parametric-Contrastive-Learning

    default params is a default hparams of cifar10/cifar100 in
        Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition.
        see details in https://github.com/Megvii-Nanjing/BBN/blob/7992e90884/configs/cifar100.yaml

    Args:
        labels:
        imb_type:
        imb_factor:

    Returns:

    """
    labels = np.array(labels)
    cats = set(labels)
    cat_num = len(cats)
    img_max = len(labels) // cat_num
    if imb_type == 'exp':
        img_num_per_cls = [int(img_max * imb_factor ** (i / (cat_num - 1.0))) for i in range(cat_num)]
    elif imb_type == 'step':
        half = cat_num // 2
        img_num_per_cls = [img_max] * half + [int(img_max * imb_factor)] * (cat_num - half)
    else:
        img_num_per_cls = [img_max] * cat_num

    indexs = []
    for i, num in zip(range(cat_num), img_num_per_cls):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        indexs.extend(idx[:num])
    return indexs
