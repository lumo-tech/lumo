import sqlite3
import json
import sys
import os
from collections import namedtuple
from lumo.utils.paths import cache_dir
from lumo.utils.hash import string_hash
from lumo.base_classes import attr

row = namedtuple('row', ['id', 'value'])


class _PopQueue():
    def __init__(self, ids, records):
        self.ids = ids
        self.records = records
        self.i = 0
        self.idset = set()

    def update(self, ids, records):
        nids = []
        nrecs = []
        for id, record in zip(ids, records):
            if id in self.idset:
                continue
            nids.append(id)
            nrecs.append(record)

        self.i = 0
        self.ids = nids
        self.records = nrecs
        if len(nids) > 0:
            return True
        else:
            return False

    def has_next(self):
        return self.i < len(self.records)

    def top(self):
        return row(self.ids[self.i], self.records[self.i])

    def pop(self):
        if not self.has_next():
            return None

        res = self.top()
        self.idset.add(res[0])
        self.i += 1
        return res

    def __len__(self):
        return len(self.ids)


class MPStruct:
    TABLE_NAME = None
    TABLE_SQL = None
    INDEX_SQL = None

    def __init__(self, session_id=None, root=None, retry=50):
        if session_id is None:
            session_id = string_hash(sys.argv[0])

        if root is None:
            root = os.path.join(cache_dir(), '.mpqueue')
            os.makedirs(root, exist_ok=True)
        self.file = os.path.join(root, f"{self.__class__.__name__}_{session_id}.sqlite")
        self.root = root
        self._retry = retry
        self.session_id = session_id
        self._connect = None
        self._initialized = False
        self._tables = None
        self._pop_queue = None

    @property
    def connect(self):
        if self._connect is None:
            self._connect = sqlite3.connect(self.file)
        return self._connect

    def reconnect(self):
        self._connect.close()
        self._connect = sqlite3.connect(self.file)

    @property
    def cursor(self):
        if not self._initialized:
            self.init_table()
            self._initialized = True
        return self.connect.cursor()

    def init_table(self):
        raise NotImplemented()

    def has_table(self, key):
        if self._tables is None:
            conn = self.connect
            c = conn.cursor()
            self._tables = {i[0] for i in
                            c.execute("""SELECT name FROM sqlite_master WHERE type='table';""").fetchall()}
        return key in self._tables

    def encode_value(self, value):
        res = attr()
        res.value = value
        return json.dumps(res.jsonify())

    def decode_value(self, res):
        res = json.loads(res)
        return attr.from_dict(res).value

    def execute(self, sql, mode='r'):
        if mode == 'r':
            res = self.cursor.execute(sql)
            return res
        else:
            for i in range(self._retry):
                try:
                    res = self.cursor.execute(sql)
                    self.connect.commit()
                    return res
                except sqlite3.OperationalError as e:
                    from lumo.kit.logger import get_global_logger
                    get_global_logger().debug(f'[mpqueue] retry {i:02d}/{self._retry}...')
                    continue
        return None


class Queue(MPStruct):
    TABLE_NAME = 'QUEUE'
    TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS QUEUE
           (ID      INTEGER PRIMARY KEY autoincrement,
           VALUE    TEXT    NOT NULL);
        """

    INDEX_SQL = """
           CREATE UNIQUE INDEX QVALUE_INDEX on QUEUE (VALUE)
    """

    def init_table(self):
        if self.has_table(self.TABLE_NAME):
            return

        conn = self.connect
        c = conn.cursor()
        c.executescript('\n'.join([self.TABLE_SQL]))

        try:
            c.execute(self.INDEX_SQL)
        except:
            pass
        conn.commit()

    def _del_rec(self, id, table):
        res = self.execute(f"delete from {table} where id={id};", 'w')
        return res is not None and res.rowcount > 0

    def push(self, value):
        if self.value_in_queue(value) is not None:
            return False
        value = self.encode_value(value)

        return self.execute(f"insert into queue (value) values ('{value}');", 'w') is not None

    @property
    def pop_queue(self):
        if self._pop_queue is None or not self._pop_queue.has_next():
            recs = self.cursor.execute("select id,value from queue;").fetchall()
            ids = [i[0] for i in recs]
            records = [self.decode_value(i[1]) for i in recs]
            if self._pop_queue is None:
                self._pop_queue = _PopQueue(ids, records)
            else:
                self._pop_queue.update(ids, records)
        return self._pop_queue

    def pop(self):
        rec = self.execute(f"select id,value from queue limit 1;").fetchone()
        if rec is not None:
            id, value = rec
            value = self.decode_value(value)
            if not self._del_rec(id, self.TABLE_NAME):
                return self.pop()
            return row(id, value)
        return None

    def value_in_queue(self, value):
        value = self.encode_value(value)
        res = self.execute(f"select id from queue where value='{value}';").fetchone()
        if res is not None:
            return res[0]
        else:
            return None

    def top(self):
        return self.pop_queue.top()

    @property
    def count(self):
        return self.execute('select count(id) from queue').fetchone()[0]


class Marker(MPStruct):

    def mark(self, key, tag):
        pass

    def unmark(self, key, tag):
        pass

    def has_marks(self, key, *tag):
        pass

    def get_marks(self, key):
        pass


class Dict(MPStruct):
    TABLE_NAME = 'DICT'
    TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS DICT
       (ID      INTEGER PRIMARY KEY autoincrement,
       KEY      TEXT    NOT NULL,
       VALUE    TEXT    NOT NULL);
    CREATE UNIQUE INDEX KEY_INDEX on DICT (KEY);
       '''

    def init_table(self):
        if self.has_table(self.TABLE_NAME):
            return

        conn = self.connect
        c = conn.cursor()
        c.executescript('\n'.join([self.TABLE_SQL]))
        conn.commit()

    def set(self, key, value: str):
        res = self.get(key)
        value = self.encode_value(value)

        if res is None:
            self.execute(f"insert into dict (key,value) values ('{key}','{value}');")

    def get(self, key, default=None):
        res = self.execute(f"select id,value from dict where key='{key}';").fetchone()
        if res is None:
            return default
        else:
            return row(res[0], self.decode_value(res[1]))


if __name__ == '__main__':
    print()
