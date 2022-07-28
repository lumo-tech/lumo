import os.path

from dbrecord import PDict
from lumo.proc import path
from lumo.utils.filelock import Lock


class Table:
    pass


class TableRow:
    def __init__(self, table, rowkey):
        dirpath = os.path.join(path.libhome(), 'database')
        os.makedirs(dirpath, exist_ok=True)
        self.fpath = os.path.join(dirpath, f'{table}.sqlite')
        self.key = rowkey
        self.value = {}
        self.disk = PDict(self.fpath)
        self.lock = Lock(table)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def flush(self):
        with self.lock:
            self.disk[self.key] = self.value
            self.disk.flush()

    def update_metrics(self, dic: dict, compare=None, flush=False):
        res = {}
        [res.update(self.update_metric(k, v, compare)) for k, v in dic.items()]

        if flush:
            self.flush()
        return res

    def update_metric(self, key, value, compare=None, flush=False):
        dic = self.value.setdefault('metric', {})
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

        return {key: old}

    def update_metric_pair(self, key, value, key2, value2, compare=None, flush=False):
        dic = self.value.setdefault('metric', {})
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

        return {key: old, key2: old2}

    def set_params(self, params: dict):
        self.value['params'] = params
        self.flush()
        return params

    def update_dict(self, dic: dict, flush=True):
        for k, v in dic.items():
            self.update(k, v)
        if flush:
            self.flush()

    def update(self, key, value, flush=True):
        self.value[key] = value
        if flush:
            self.flush()
