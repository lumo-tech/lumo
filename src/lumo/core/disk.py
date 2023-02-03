import os.path
from dbrecord import PList
from lumo.proc import path
from lumo.utils.filelock2 import Lock
from lumo.utils import safe_io as IO


class Metrics:
    """
    Record metrics at multiple steps. Supported by dbrecord.
    """

    def __init__(self, test_path: str):
        os.makedirs(test_path, exist_ok=True)
        self.fpath = os.path.join(test_path, f'metric_board.sqlite')
        self.disk = PList(self.fpath)
        self.lock = Lock(os.path.basename(test_path.rstrip('/')))

    def append(self, metric: dict, step, stage='train'):
        self.disk.append({
            'metric': metric,
            'step': step,
            'stage': stage
        })
        self.disk.flush()

    def flush(self):
        self.disk.flush()


class TableRow:
    """
    It can be regarded as a serialized dictionary,
    or a certain row in the table, so the same key value will be overwritten.

    If you need to record records at different times, please use trainer.metrics
    """

    def __init__(self, table, partition, rowkey):
        dirpath = os.path.join(path.libhome(), 'metrics', table)
        os.makedirs(dirpath, exist_ok=True)
        self.fpath = os.path.join(dirpath, partition, f'{rowkey}.pkl')
        self.key = rowkey
        self.value = {}
        # self.disk = PDict(self.fpath)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def flush(self):
        IO.dump_pkl(self.value, self.fpath)

    def update_metrics(self, dic: dict, compare=None, flush=False):
        res = {}
        [res.update(self.update_metric(k, v, compare)) for k, v in dic.items()]

        if flush:
            self.flush()
        return res

    def update_metric(self, key, value, compare=None, flush=False):
        dic = self.metric
        old = dic.setdefault(key, None)

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
                assert False

        if update:
            dic[key] = value
            old = value

        if flush:
            self.flush()

        return {key: value}

    @property
    def metric(self):
        return self.value.setdefault('metric', {})

    def update_metric_pair(self, key, value, key2, value2, compare=None, flush=False):
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
                assert False

        if update:
            dic[key] = value
            dic[key2] = value2
            old, old2 = value, value2

        if flush:
            self.flush()

        return {key: value, key2: value2}

    def set_params(self, params: dict):
        self.value['params'] = params
        self.flush()
        return params

    def update_dict(self, dic: dict, flush=False):
        for k, v in dic.items():
            self.update(k, v)
        if flush:
            self.flush()

    def update(self, key, value, flush=True):
        self.value[key] = value
        if flush:
            self.flush()
