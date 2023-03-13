import os
from lumo.utils import safe_io as IO


class Metric:
    """
    """

    def __init__(self, metric_fn, persistent=True):
        os.makedirs(os.path.dirname(os.path.abspath(metric_fn)), exist_ok=True)
        self.fn = metric_fn
        self._metric = {}
        if os.path.exists(metric_fn):
            self._metric = IO.load_pkl(metric_fn)
        self.persistent = persistent

    @property
    def value(self):
        """
        A property that returns the metric values of the row.

        Returns:
            dict: A dictionary containing the metric values of the row.
        """
        return self._metric

    def dump_metric(self, key, value, cmp: str, flush=True, **kwargs):
        dic = self.value
        older = dic.setdefault(key, None)

        update = False
        if older is None or cmp is None:
            update = True
        else:
            if cmp == 'max':
                if older < value:
                    update = True
            elif cmp == 'min':
                if older > value:
                    update = True
            else:
                raise NotImplementedError()

        if update:
            dic[key] = value
            for kk, vv in kwargs.items():
                dic[kk] = vv

        if flush:
            self.flush()
        return value

    def dump_metrics(self, dic: dict, cmp: str):
        for k, v in dic.items():
            self.dump_metric(k, v, cmp)

    def flush(self):
        """Writes the value of the row to a file."""
        if self.persistent:
            IO.dump_pkl(self.value, self.fn)
