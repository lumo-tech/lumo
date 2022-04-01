"""
for calculating and recording accuracy.
"""
import torch

from lumo import Meter, Params
try:
    import sklearn
except ImportError:
    import warnings
    warnings.warn("You need to install scikit-learn to use UnsupervisedAccMixin class ")

class AccMixin():
    pass


class UnsupervisedAccMixin(AccMixin):
    def acc_assigned_(self):
        pass  # TODO

    def acc_nmi_(self, pred_labels: torch.Tensor, labels: torch.Tensor,
                 meter: Meter = None, name='nmi'):
        """
        Normalized Mutual Information for Unsupervised
        Args:
            pred_labels: (batchsize, )
            labels: (batchsize, )
            meter:
            name:

        Returns:
            nmi
        """
        from sklearn import metrics
        y_true = labels.cpu().numpy()
        y_pred = pred_labels.cpu().numpy()

        acc = metrics.normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        if meter is not None:
            meter[name] = acc
        return acc

    def acc_ari_(self, pred_labels: torch.Tensor, labels: torch.Tensor,
                 meter: Meter = None, name='ari'):
        """
        Rand index adjusted for Unsupervised
        Args:
            pred_labels: (batchsize, )
            labels: (batchsize, )
            meter:
            name:

        Returns:
            nmi
        """
        from sklearn import metrics

        y_true = labels.cpu().numpy()
        y_pred = pred_labels.cpu().numpy()

        acc = metrics.adjusted_rand_score(y_true, y_pred)
        if meter is not None:
            meter[name] = acc
        return acc

    def test_eval_logic(self, dataloader, param: Params):
        from sklearn import metrics
        import numpy as np
        with torch.no_grad():
            meter = Meter()
            y_trues = []
            y_preds = []
            for xs, y_true in dataloader:
                preds = self.predict(xs)  # type:torch.Tensor

                y_pred = preds.argmax(dim=-1).cpu().numpy()
                y_true = y_true.cpu().numpy()

                y_preds.extend(y_pred)
                y_trues.extend(y_true)

            meter.nmi = metrics.normalized_mutual_info_score(
                np.array(y_trues), np.array(y_preds),
                average_method="arithmetic")

            meter.ari = metrics.adjusted_rand_score(np.array(y_trues), np.array(y_preds))

        return meter


class ClassifyAccMixin(AccMixin):
    def acc_precise_(self, pred_labels: torch.Tensor, labels, meter: Meter = None, name='acc'):
        """train batch accuracy"""
        with torch.no_grad():
            maxid = pred_labels
            total = labels.shape[0]
            top1 = (labels == maxid).sum().float() / total
            top1 = top1.item()

        if meter is not None:
            meter[name] = top1
            meter.percent(name)
        return top1

    def test_eval_logic(self, dataloader, param: Params):
        from lumo.contrib.torch import accuracy as acc
        param.topk = param.default([1, 5])
        with torch.no_grad():
            count_dict = Meter()
            for xs, labels in dataloader:
                preds = self.predict(xs)
                total, topk_res = acc.classify(preds, labels, topk=param.topk)
                count_dict["total"] += total
                for i, topi_res in zip(param.topk, topk_res):
                    count_dict["top{}".format(i)] += topi_res
        return count_dict
