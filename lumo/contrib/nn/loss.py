import torch
from .functional import normalize, masked_log_softmax
from torch.nn import functional as F


def contrastive_loss(query: torch.Tensor, key: torch.Tensor,
                     norm=False,
                     temperature=0.7,
                     inbatch_neg=False,
                     graph_mask=None,
                     ):
    """

    Examples:
        from lumo.contrib.nn import functional as LF
        logits = torch.rand(4,128)
        logits_s = torch.rand(4,128)
        contrastive_loss(logits,logits_s,norm=True,inbatch_neg=True)

        logits = LF.normalize(logits)
        logits_s = LF.normalize(logits_s)
        contrastive_loss(logits,logits_s,norm=False,inbatch_neg=True)

        # sanity check
        # see `InstanceLoss` from https://github.com/Yunfan-Li/Contrastive-Clustering/blob/main/modules/contrastive_loss.py
        loss_fn = InstanceLoss(4,0.7,'cpu')
        loss_fn(logits,logits_s)
        contrastive_loss(logits,logits_s,inbatch_neg=True,temperature=0.7)

    Args:
        query: [Bq, feature_dim]
        key: [Bk, feature_dim]
        norm: bool, norm or not
        temperature: float, temperature
        inbatch_neg: bool,
        graph_mask: [Bq, Bk] or [Bq+Bk, Bq+Bk] if inbatch_neg = True

    Returns:

    """
    # if norm:
    #     query = normalize(query, dim=-1)
    #     key = normalize(key, dim=-1)

    qbs = len(query)

    if inbatch_neg:
        query = torch.cat([query, key])
        key = query
    if norm:
        logits = torch.cosine_similarity(query.unsqueeze(1), key.unsqueeze(0), dim=2)
    else:
        logits = torch.mm(query, key.t())  # query @ key.t()

    # apply temperature
    logits /= temperature

    # make label
    if graph_mask is None:
        graph_mask = torch.eye(len(logits), dtype=torch.float, device=query.device)
        if inbatch_neg:
            graph_mask = torch.cat([graph_mask[qbs:], graph_mask[:qbs]])

    # make same sample mask
    smask = None
    if inbatch_neg:
        smask = (1 - torch.eye(len(logits), dtype=torch.float, device=query.device))
        graph_mask = (graph_mask * smask)

    loss = -torch.sum(masked_log_softmax(logits, smask, dim=-1) * graph_mask, dim=1)
    loss = (loss / graph_mask.sum(1)).mean()  # when self-supervised without cluster, it equals to loss.mean()
    return loss


def cluster_loss(query, key, label_graph_mask=None, temperature=0.7):
    """
    see https://github.com/Yunfan-Li/Contrastive-Clustering

    Examples:
        assert (prob_a.sum(dim=-1) == 1).all()
        assert (prob_b.sum(dim=-1) == 1).all()
        cluster_loss(prob_a,prob_b)

        # sanity check
        # see `ClusterLoss` from https://github.com/Yunfan-Li/Contrastive-Clustering/blob/main/modules/contrastive_loss.py
        loss_fn = ClusterLoss(5,0.7,'cpu')
        loss_fn(prob_a,prob_b)
        cluster_loss(prob_a,prob_b,temperature=0.7)


    """
    return contrastive_loss(query.t(), key.t(), norm=True,
                            temperature=temperature,
                            inbatch_neg=True,
                            graph_mask=label_graph_mask)


def sup_contrastive_loss(query: torch.Tensor, key: torch.Tensor,
                         label=None,
                         norm=False,
                         temperature=0.7):
    """
    A modified version from https://github.com/HobbitLong/SupContrast

    Examples:
        q = torch.rand(4,128)
        k = torch.rand(4,128)
        label = torch.randint(0,10,(4,))
        sup_contrastive_loss(q,k,label,norm=True)

        # sanity check
        sup = SupConLoss()
        sup(torch.stack([a,b],1))
        sup_contrastive_loss(a,b,norm=False,temperature=0.07)

    """
    if label is None:
        graph = None
    else:
        label = label.unsqueeze(0)
        graph = (label == label.T)
        graph = graph.repeat(2, 2)
    return contrastive_loss(
        query, key,
        norm=norm, temperature=temperature, inbatch_neg=True,
        graph_mask=graph,
    )


def cross_entropy_with_targets(logits, targets, mask=None):
    """

    Args:
        logits: [bs, class_num]
        targets: [bs, class_num]
        mask: [bs,]

    Returns:

    """
    out = torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1)
    if mask is not None:
        out = out * mask.float()
    return -torch.mean(out)
