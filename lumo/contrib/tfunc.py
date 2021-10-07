import torch
from torch.nn import functional as F


def normalize(feature: torch.Tensor, inplace=False, eps=1e-08):
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
