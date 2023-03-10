import os.path
import warnings

from dbrecord import PList

from lumo.proc import path
from lumo.utils import safe_io as IO
from lumo.decorators.deprecated import DeprecatedWarning


class Metrics:
    """
    Records metrics at multiple steps and stages. The metrics are supported by dbrecord.

    Args:
        test_path (str): The path to the test directory.

    Attributes:
        fpath (str): The path to the metric board SQLite file.
        disk (PList): The PList instance for accessing the metric board SQLite file.

    Methods:
        append(metric: dict, step: int, stage: str = 'train') -> None:
            Adds the specified metric, step, and stage to the metric board SQLite file.
            The metric is a dictionary object that contains the metric name as the key and the metric value as the value.
            The stage is either 'train' or 'test', and it is set to 'train' by default.
            This method calls the `flush` method to write the changes to disk.

        flush() -> None:
            Writes any changes to the metric board SQLite file to disk.
    """

    def __init__(self, test_path: str, persistent=True):
        os.makedirs(test_path, exist_ok=True)
        self.fpath = os.path.join(test_path, f'metric_board.sqlite')
        self.disk = PList(self.fpath)
        self.persistent = persistent

    def append(self, metric: dict, step, stage='train'):
        """
        Adds the specified metric, step, and stage to the metric board SQLite file.

        Args:
            metric (dict): A dictionary object that contains the metric name as the key and the metric value as the value.
            step (int): The step number.
            stage (str, optional): The stage of the metric, either 'train' or 'test'. Defaults to 'train'.

        Returns:
            None
        """
        self.disk.append({
            'metric': metric,
            'step': step,
            'stage': stage
        })
        self.disk.flush()

    def flush(self):
        """
        Writes any changes to the metric board SQLite file to disk.

        Returns:
            None
        """
        if self.persistent:
            self.disk.flush()


class TableRow:
    """
    TableRow class is a serialized dictionary that can represents a single row in a table.
    If the same key is updated, its value will be overwritten.
    Please use trainer.metrics to record records at different times.

    Args:
    - table (str): name of the table to which the row belongs.
    - partition (str): partition of the row.
    - rowkey (str): unique identifier of the row.

    Attributes:
    - fpath (str): path of the file that stores the serialized row.
    - key (str): unique identifier of the row.
    - value (dict): dictionary representing the row.
    - persistent (bool): whether to store in disk.

    Methods:
    - __enter__(self): context manager method. Does nothing.
    - __exit__(self, exc_type, exc_val, exc_tb): context manager method. Calls flush method.
    - flush(self): writes the value of the row to a file.
    - update_metrics(self, dic: dict, compare=None, flush=False): updates multiple metrics in the row.
    - update_metric(self, key, value, compare=None, flush=False): updates a single metric in the row.
    - metric(self): returns the metric dictionary of the row.
    - update_metric_pair(self, key, value, key2, value2, compare=None, flush=False): updates two metrics in the row.
    - set_params(self, params: dict): sets the value of 'params' key in the row.
    - update_dict(self, dic: dict, flush=False): updates multiple keys in the row.
    - update(self, key, value, flush=True): updates a single key in the row.
    - __getitem__(self, item): returns the value of a key in the row.
    """

    def __init__(self, fn, persistent=True):
        os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
        self.fpath = fn
        self.value = {}
        self.persistent = persistent

        # self.disk = PDict(self.fpath)

    def __enter__(self):
        """
        Does nothing.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Calls flush method. Required for using the object as a context manager.
        """
        self.flush()

    def flush(self):
        """Writes the value of the row to a file."""
        if self.persistent:
            IO.dump_pkl(self.value, self.fpath)

    def update_metrics(self, dic: dict, compare=None, flush=False):
        """
        Updates multiple metrics in the row.

        Args:
        - dic (dict): dictionary containing key-value pairs to be updated.
        - compare (str): comparison operator to be used for updating metrics. Only 'max' and 'min' are supported.
        - flush (bool): if True, writes the value of the row to a file after updating the metrics.

        Returns:
        - res (dict): dictionary containing key-value pairs that were updated.
        """
        res = {}
        [res.update(self.update_metric(k, v, compare)) for k, v in dic.items()]

        if flush:
            self.flush()
        return res

    def update_metric(self, key, value, compare=None, flush=False):
        """
        Updates a metric value in the row.

        Args:
            key (str): The key of the metric.
            value (float): The value of the metric.
            compare (str, optional): The comparison operator used to compare the new value with the old one.
                Either 'max' or 'min'. Default is None.
            flush (bool, optional): Whether to flush the changes to disk. Default is False.

        Returns:
            dict: A dictionary containing the updated metric key and value.
        """
        dic = self.metric
        older = dic.setdefault(key, None)

        update = False
        if older is None or compare is None:
            update = True
        else:
            if compare == 'max':
                if older < value:
                    update = True
            elif compare == 'min':
                if older > value:
                    update = True
            else:
                raise NotImplementedError()

        if update:
            dic[key] = value
            older = value

        if flush:
            self.flush()

        return {key: older}

    @property
    def metric(self):
        """
        A property that returns the metric values of the row.

        Returns:
            dict: A dictionary containing the metric values of the row.
        """
        return self.value.setdefault('metric', {})

    def update_metric_pair(self, key, value, key2, value2, compare=None, flush=False):
        """
        Update a pair of key-value metrics in the metric dictionary.

        Args:
            key (str): The key of the first metric.
            value (float): The value of the first metric.
            key2 (str): The key of the second metric.
            value2 (float): The value of the second metric.
            compare (str, optional): The method to compare values. Default is None.
                                     Possible values are 'max', 'min'.
            flush (bool, optional): Whether to flush to disk after updating. Default is False.

        Returns:
            dict: A dictionary with the old values of the updated metrics.
        """
        dic = self.metric
        old = dic.setdefault(key, None)
        old2 = dic.setdefault(key2, None)

        update = False
        if old is None or compare is None:
            update = True
        else:
            if compare == 'max':
                if old < value:
                    update = True
            elif compare == 'min':
                if old > value:
                    update = True
            else:
                raise NotImplementedError()

        if update:
            dic[key] = value
            dic[key2] = value2
            old, old2 = value, value2

        if flush:
            self.flush()

        return {key: old, key2: old2}

    def __getitem__(self, item):
        """Get the value of a key in the row."""
        return self.value[item]


DeprecatedWarning(TableRow, '0.15.0', '1.0.0',
                  'This class is deprecated and will be remove in 1.0.0, '
                  'Please use Experiment.metric to record your best metric.')
