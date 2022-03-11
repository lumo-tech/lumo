from typing import Optional

import torch
from torch.nn import functional as F

from lumo.contrib.nn.functional import masked_log_softmax


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


def contrastive_loss2(query: torch.Tensor, key: torch.Tensor,
                      memory: Optional[torch.Tensor] = None,
                      norm: Optional[bool] = False,
                      temperature: float = 0.7,
                      query_neg: Optional[bool] = False,
                      key_neg: Optional[bool] = True,
                      qk_graph: Optional[torch.Tensor] = None,
                      qm_graph: Optional[torch.Tensor] = None,
                      eye_one_in_qk: Optional[bool] = False,
                      softmax_mask=None,
                      reduction: Optional[str] = 'mean'
                      ):
    """
    Examples:
        >>> assert contrastive_loss(a,b,inbatch_neg=False) == contrastive_loss2(a,b,query_neg=False,key_neg=True)

    Args:
        query: [bq, f_dim]
        key: [bq, f_dim]
        memory: [bm, f_dim]
        norm: bool, normalize feature or not
        temperature: float
        query_neg: bool
        key_neg: bool
        qk_graph: [bq, bq], >= 0
        qm_graph: [bq, bm], >= 0
        eye_one_in_qk: bool
        reduction: str
            'mean',
            'sum',
            'none'

    Returns:
        loss

    """
    if memory is None:
        key_neg = True
        qm_graph = None

    if memory is not None:
        key = torch.cat([key, memory])

    if query_neg:
        key = torch.cat([query, key])

    q_size = query.shape[0]

    if norm:
        logits = torch.cosine_similarity(query.unsqueeze(1), key.unsqueeze(0), dim=2)
    else:
        logits = torch.mm(query, key.t())  # query @ key.t()

    logits /= temperature

    neg_index = torch.ones_like(logits, dtype=torch.bool, device=logits.device)

    _temp_eye_indice = torch.arange(q_size, device=logits.device).unsqueeze(1)

    neg_offset = q_size if query_neg else 0
    if query_neg:
        neg_index.scatter_(1, _temp_eye_indice, 0)

    if key_neg:
        neg_index.scatter_(1, _temp_eye_indice + neg_offset, 0)
    else:
        neg_index[:, neg_offset:neg_offset + q_size] = 0

    pos_index = torch.zeros_like(logits, dtype=torch.float, device=logits.device)
    pos_offset = q_size if query_neg else 0

    # for supervised cs
    if qk_graph is not None:
        pos_index[:, :q_size] = qk_graph.float()
        if query_neg:
            pos_index[:, q_size:q_size * 2] = qk_graph.float()

    # for supervised cs with moco memory bank
    if qm_graph is not None:
        pos_index[:, pos_offset + q_size:] = qm_graph.float()

    if qk_graph is None or eye_one_in_qk:
        pos_index.scatter_(1, _temp_eye_indice + pos_offset, 1)

    logits_mask = (pos_index > 0) | neg_index
    if softmax_mask is not None:
        if logits_mask.shape != softmax_mask.shape:
            raise ValueError(f'softmax_mask.shape should be {logits_mask.shape}, but got {softmax_mask.shape}')
        logits_mask = softmax_mask.bool() * logits_mask
    loss = -torch.sum(masked_log_softmax(logits, logits_mask, dim=-1) * pos_index, dim=1)
    loss = (loss / pos_index.sum(1))
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

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


def minent(logits, w_ent=1):
    loss = - torch.sum(F.log_softmax(logits, dim=-1) * F.softmax(logits, dim=-1), dim=-1).mean() * w_ent
    return loss
