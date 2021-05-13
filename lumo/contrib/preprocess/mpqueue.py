import sqlite3
import json
import sys
import os
from collections import namedtuple
from lumo.utils.paths import cache_dir
from lumo.utils.hash import string_hash
from lumo.base_classes import attr

_DICT_TABLE = '''
    CREATE TABLE IF NOT EXISTS DICT
       (ID      INTEGER PRIMARY KEY autoincrement,
       KEY      TEXT    NOT NULL,
       VALUE    TEXT    NOT NULL);
    CREATE UNIQUE INDEX KEY_INDEX on DICT (KEY);
       '''

_QUEUE_TABLE = """
    CREATE TABLE IF NOT EXISTS QUEUE
           (ID      INTEGER PRIMARY KEY autoincrement,
           VALUE    TEXT    NOT NULL);
   CREATE UNIQUE INDEX QVALUE_INDEX on QUEUE (VALUE);
        """

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


class Queue:
    def __init__(self, session_id=None, root=None):
        if session_id is None:
            session_id = string_hash(sys.argv[0])

        if root is None:
            root = os.path.join(cache_dir(), '.mpqueue')

        self.file = os.path.join(root, f"{session_id}.sqlite")
        self.root = root
        self.session_id = session_id
        self._connect = None

        self._pop_queue = None

    @property
    def connect(self):
        if self._connect is None:
            self._connect = sqlite3.connect(self.file)
        return self._connect

    @property
    def cursor(self):
        self._init_table()
        self.connect.commit()
        return self.connect.cursor()

    def _init_table(self):
        conn = self.connect
        c = conn.cursor()

        res = {i[0] for i in c.execute("""SELECT name FROM sqlite_master WHERE type='table';""").fetchall()}
        if 'DICT' in res and 'QUEUE' in res:
            return

        c.executescript('\n'.join([_DICT_TABLE, _QUEUE_TABLE]))
        conn.commit()

    def _encode_value(self, value):
        res = attr()
        res.value = value
        return json.dumps(res.jsonify())

    def _decode_value(self, res):
        res = json.loads(res)
        return attr.from_dict(res).value

    def _del_rec(self, id, table):
        self.cursor.execute(f"delete from {table} where id={id};")

    def set(self, key, value: str):
        res = self.get(key)
        value = self._encode_value(value)

        if res is None:
            self.cursor.execute(f"insert into dict (key,value) values ('{key}','{value}');")

    def get(self, key, default=None):
        res = self.cursor.execute(f"select id,value from dict where key='{key}';").fetchone()
        if res is None:
            return default
        else:
            return row(res[0], self._decode_value(res[1]))

    def push(self, value):
        if self.value_in_queue(value) is not None:
            return False
        value = self._encode_value(value)

        self.cursor.execute(f"insert into queue (value) values ('{value}');")
        return True

    @property
    def pop_queue(self):
        if self._pop_queue is None or not self._pop_queue.has_next():
            recs = self.cursor.execute("select id,value from queue;").fetchall()
            ids = [i[0] for i in recs]
            records = [self._decode_value(i[1]) for i in recs]
            if self._pop_queue is None:
                self._pop_queue = _PopQueue(ids, records)
            else:
                self._pop_queue.update(ids, records)
        return self._pop_queue

    def pop(self):
        return self.pop_queue.pop()

    def value_in_queue(self, value):
        value = self._encode_value(value)
        res = self.cursor.execute(f"select id from queue where value='{value}';").fetchone()
        if res is not None:
            return res[0]
        else:
            return None

    def top(self):
        return self.pop_queue.top()

    def mark(self, key, tag):
        pass

    def unmark(self, key, tag):
        pass

    def has_marks(self, key, *tag):
        pass

    def get_marks(self, key):
        pass


if __name__ == '__main__':
    print()
