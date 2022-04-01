"""

"""
from typing import Tuple, List, Callable

import torch


def rotate_right_angle(x: torch.Tensor, w_dim: int = 2, h_dim: int = 3, degree: int = 90):
    assert degree in {90, 270, 180}
    if degree == 90:
        x = x.transpose(w_dim, h_dim)  # 90
    elif degree == 180:
        x = x.flip(w_dim)
    elif degree == 270:
        x = x.transpose(w_dim, h_dim).flip(h_dim)  # 270

    return x


def split_sub_matrix(mat: torch.Tensor, *sizes):
    """
    将一个[N,M,...,L]的矩阵按 n,m,...l 拆分成 N/n*M/m*...L/l 个 [n,m,...l]的小矩阵

    如果N/n 无法整除，不会报错而是会将多余的裁掉
    example:
        mat = torch.arange(0,24).view(4,6) # shape = [4, 6]
        >> tensor([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11],
                   [12, 13, 14, 15, 16, 17],
                   [18, 19, 20, 21, 22, 23]])

        split_sub_matrix(mat,2,3) # shape = [2, 2, 2, 3]
        >> tensor([[[[ 0,  1,  2],
                      [ 6,  7,  8]],

                     [[ 3,  4,  5],
                      [ 9, 10, 11]]],

                    [[[12, 13, 14],
                      [18, 19, 20]],

                     [[15, 16, 17],
                      [21, 22, 23]]]])

    :param mat: 一个[N,M,...L] 的矩阵
    :param sizes: n,m,...l 的list, 其长度不一定完全和mat的维数相同
        mat = torch.arange(0,240).view([4,6,10])
        split_sub_matrix(mat,2,3) # shape = [2, 2, 10, 2, 3]
    :return: 一个 [N/row,M/col,row,col] 的矩阵
    """
    for i, size in enumerate(sizes):
        mat = mat.unfold(i, size, size)
    return mat


def onehot(labels: torch.Tensor, label_num):
    """
    convert label to onehot vector
    Args:
        labels:
        label_num:

    Returns:

    """
    return torch.zeros(*labels.shape, label_num, device=labels.device).scatter_(-1, labels.unsqueeze(-1), 1)


def cartesian_product(left: torch.Tensor, right: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate cartesian product of the given tensor(s)

    Args:
        left: A pytorch tensor.
        right: A pytorch tensor,
            if None, wile be left X left

    Returns:
        Tuple[torch.Tensor, torch.Tensor]

    Example:
        >>> cartesian_product(torch.arange(0,3),torch.arange(0,5))

    (tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
      tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))

    """
    if right is None:
        right = left

    nleft = left.repeat_interleave(right.shape[0], dim=0)
    nright = right.repeat(*[item if i == 0 else 1 for i, item in enumerate(left.shape)])
    return nleft, nright


def cat_then_split(op: Callable[[torch.Tensor], torch.Tensor], tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Examples:
    >>> ta,tb,tc = cat_then_split(lambda x:model(x),[ta,tb,tc])
    >>> alogits,blogits,clogits = cat_then_split(model,[ta,tb,tc])
    """

    res = op(torch.cat(tensors))  # type: torch.Tensor
    return res.split_with_sizes([i.shape[0] for i in tensors])


def label_smoothing(onehot_labels, epsilon=0.1):
    """
    Applies label smoothing


    Args:
        onehot_labels:
        epsilon:

    Returns:

    References:
        see
        https://arxiv.org/abs/1512.00567
        https://arxiv.org/abs/1906.02629
    """
    return ((1 - epsilon) * onehot_labels) + (epsilon / onehot_labels.shape[-1])


def label_smoothing_v2(onehot_labels, rate=0.1):
    """
    Applies label smoothing


    Args:
        onehot_labels:
        epsilon:

    Returns:

    References:
        see
        https://arxiv.org/abs/1512.00567
        https://arxiv.org/abs/1906.02629
    """
    mixed = torch.ones_like(onehot_labels) * (1 / onehot_labels.shape[-1])
    return ((1 - rate) * onehot_labels) + rate * mixed


def elementwise_mul(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiplication of a vector and a matrix

    References:
        https://discuss.pytorch.org/t/element-wise-multiplication-of-a-vector-and-a-matrix/56946

    Args:
        vec: tensor of shape (N, )
        mat: matrix of shape (N, ...)

    Returns:
        A tensor

    """
    nshape = [-1] + [1] * (len(mat.shape) - 1)
    nvec = vec.reshape(nshape)  # .expand_as(mat)
    return nvec * mat


def euclidean_dist(x, y, min=1e-12):
    """
    copy from https://blog.csdn.net/IT_forlearn/article/details/100022244
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist = dist - 2 * torch.mm(x, y.T)
    # dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=min).sqrt()  # for numerical stability
    return dist


def label_eq_matric(labels: torch.Tensor):
    """

    Args:
        labels: ground truth of shape [bsz].

    Returns:
        a bool tensor [bsz, bsz],
    """
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T)

    return mask


def resize(arr, newsize, mode='bilinear'):
    pass  # TODO
