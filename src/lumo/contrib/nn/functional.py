import torch
from torch.nn import functional as F

import torch
from torch.nn import functional as F


def normalize(feature: torch.Tensor, inplace=False, eps=1e-08):
    """
    a = torch.rand(2, 3)
    b = torch.rand(2, 3)
    sim_1 = batch_cosine_similarity2(a, b)
    sim_2 = batch_cosine_similarity(a, b)
    print(a)
    print(b)
    print(sim_1)
    print(sim_2)
    print((sim_1 - sim_2) ** 2)
    Args:
        feature:
        inplace:
        eps:

    Returns:

    """
    if inplace:
        feature /= (feature.norm(dim=-1, keepdim=True) + eps)
    else:
        feature = feature / (feature.norm(dim=-1, keepdim=True) + eps)
    return feature


def batch_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-08):
    """
    a: [bs_a, feature_dim]
    a: [bs_b, feature_dim]
    """
    return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2, eps=eps)


def batch_cosine_similarity2(a: torch.Tensor, b: torch.Tensor, eps=0):
    """
    a: [bs_a, feature_dim]
    a: [bs_b, feature_dim]
    """

    return torch.mm(normalize(a, eps=eps), normalize(b, eps=eps).T)


def masked_log_softmax(logits, mask=None, dim=-1):
    """
    Examples:
        >>> logits = torch.rand(4,10)
        >>> log_prob = masked_log_softmax(logits)
        >>> ((logits.masked_log_softmax(dim=-1) - log_prob)**2 < 1e-07).all()

    Args:
        logits:

    Returns:

    """
    logits = logits - logits.max(dim=dim, keepdim=True).values  # avoid nan and inf when logits value is too large
    exp_logits = torch.exp(logits)
    if mask is not None:
        exp_logits = exp_logits * mask.float()
    return logits - torch.log(exp_logits.sum(dim=dim, keepdim=True))


def masked_softmax(logits, mask=None, dim=-1, eps=1e-08):
    """
    see https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
    Args:
        logits:
        mask:
        dim:
        eps:

    Returns:

    """
    if mask is None:
        return torch.softmax(logits, dim=dim)
    else:
        # matrix A is the one you want to do mask softmax at dim=1
        A_exp = torch.exp(logits - torch.max(logits, dim=1, keepdim=True).values)
        A_exp = A_exp * mask.float()  # this step masks
        res = A_exp / (torch.sum(A_exp, dim=1, keepdim=True) + eps)
        return res


def normalize(feature: torch.Tensor, inplace=False, dim=-1, eps=1e-08):
    if inplace:
        feature /= (feature.norm(dim=dim, keepdim=True) + eps)
    else:
        feature = feature / (feature.norm(dim=dim, keepdim=True) + eps)
    return feature


def batch_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-08):
    """
    a: [bs_a, feature_dim]
    a: [bs_b, feature_dim]
    """
    return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2, eps=eps)


def batch_cosine_similarity2(a: torch.Tensor, b: torch.Tensor, eps=0):
    """
    a: [bs_a, feature_dim]
    a: [bs_b, feature_dim]
    """

    return torch.mm(normalize(a, eps=eps), normalize(b, eps=eps).T)


def sharpen(x: torch.Tensor, t=1):
    """
    让概率分布变的更 sharp，即倾向于 onehot
    :param x: prediction, sum(x,dim=-1) = 1
    :param t: temperature, default is 0.5
    :return:
    """
    with torch.no_grad():
        temp = torch.pow(x, 1 / t)
        return temp / (temp.sum(dim=1, keepdim=True) + 1e-7)


if __name__ == '__main__':
    a = torch.rand(2, 3)
    b = torch.rand(2, 3)
    sim_1 = batch_cosine_similarity2(a, b)
    sim_2 = batch_cosine_similarity(a, b)
    print(a)
    print(b)
    print(sim_1)
    print(sim_2)
    print((sim_1 - sim_2) ** 2)
