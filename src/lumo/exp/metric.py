import os
from lumo.utils import safe_io as IO


class Metric:
    """
    A class that handles metric values and saving/loading them to/from disk.

    Attributes:
        fn (str): The file path of the metric file.
        _metric (dict): A dictionary containing the metric values.
        persistent (bool): A boolean value indicating whether to save the metric values to disk.
    """

    def __init__(self, metric_fn, persistent=True):
        """
        Initializes a new instance of the Metric class.

        Args:
            metric_fn (str): The file path of the metric file.
            persistent (bool): A boolean value indicating whether to save the metric values to disk.
                Default is True.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(os.path.abspath(metric_fn)), exist_ok=True)
        self.fn = metric_fn
        self._metric = {}
        self._last = {}
        if os.path.exists(metric_fn):
            self._metric = IO.load_pkl(metric_fn)

        self.persistent = persistent

    @property
    def current(self):
        return self._last

    @property
    def value(self):
        """
        A property that returns the metric values.

        Returns:
            dict: A dictionary containing the metric values.
        """
        return self._metric

    def dump_metric(self, key, value, cmp: str, flush=True, **kwargs):
        """
        Updates the metric value for a given key.

        If the metric value for the given key is not set or the new value is better than the
        existing value based on the comparison type specified by cmp, the metric value is updated
        with the new value. The function returns the updated value.

        Args:
            key (str): The key for the metric value.
            value (float): The new metric value.
            cmp (str): The type of comparison to use when updating the metric value. Must be 'max' or 'min'.
            flush (bool): A boolean value indicating whether to save the updated metric values to disk.
                Default is True.
            **kwargs: Additional key-value pairs to store with the metric value.

        Returns:
            float: The updated metric value.

        Raises:
            NotImplementedError: If cmp is not 'max' or 'min'.
        """
        dic = self._metric
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
        else:
            value = older

        self._last[key] = value
        for kk, vv in kwargs.items():
            self._last[kk] = vv

        if flush:
            self.flush()
        return value

    def dump_metrics(self, dic: dict, cmp: str):
        """
        Updates multiple metric values with a dictionary.

        The function calls dump_metric for each key-value pair in the input dictionary and returns
        a dictionary containing the updated metric values.

        Args:
            dic (dict): A dictionary containing the key-value pairs to update.
            cmp (str): The type of comparison to use when updating the metric values. Must be 'max' or 'min'.

        Returns:
            dict: A dictionary containing the updated metric values.
        """
        res = {k: self.dump_metric(k, v, cmp, flush=False)
               for k, v in dic.items()}
        self.flush()
        return res

    def flush(self):
        """
        Writes the metric values to a file.
        """
        if self.persistent:
            IO.dump_pkl(self.value, self.fn)
