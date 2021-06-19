import os
from collections import namedtuple
from lumo.base_classes.db import _NoneResult, MPStruct

row = namedtuple('row', ['id', 'value'])


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
        return not isinstance(res, _NoneResult) and res.rowcount > 0

    def _del_recs(self, *ids, table):
        ids = ','.join([str(i) for i in ids])
        res = self.execute(f"delete from {table} where id in ({ids});", 'w')
        return not isinstance(res, _NoneResult) and res.rowcount > 0

    def push(self, value):
        if self.value_in_queue(value) is not None:
            return False
        value = self.encode_value(value)

        res = self.execute(f"insert into queue (value) values ('{value}');", 'w')
        return not isinstance(res, _NoneResult)

    def pushk(self, *values):
        values = [self.encode_value(i) for i in values if self.value_in_queue(i) is None]

        if len(values) == 0:
            return 0

        inner = ','.join([f"('{value}')" for value in values])

        res = self.execute(f"insert into queue (value) values {inner};", 'w')
        if isinstance(res, _NoneResult):
            return -1
        else:
            return len(values)

    def popk(self, k=1):
        rec = self.execute(f"select id,value from queue limit {k};").fetchall()

        if len(rec) > 0:
            ids = [i[0] for i in rec]
            values = [self.decode_value(i[1]) for i in rec]

            if not self._del_recs(*ids, table=self.TABLE_NAME):
                return self.popk(k)
            return [row(id, val) for id, val in zip(ids, values)]
        return []

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
        rec = self.execute(f"select id,value from queue limit 1;").fetchone()
        if rec is not None:
            id, value = rec
            value = self.decode_value(value)
            return row(id, value)
        return None

    @property
    def count(self):
        return self.execute('select count(id) from queue').fetchone()[0]


class Bucket(Queue):

    def __init__(self, session_id=None, root=None, retry=7):
        super().__init__(session_id, root, retry)
        self.file = os.path.join(self.root, f"{Queue.__name__}_{session_id}.sqlite")

    def bucket(self, ind, total):
        rec = self.execute(f"select id,value from queue where id % {total}={ind}").fetchall()
        ids = [i[0] for i in rec]
        values = [self.decode_value(i[1]) for i in rec]
        return [row(id, val) for id, val in zip(ids, values)]

    def iter_bucket(self, ind, total):
        for rec in self.execute(f"select id,value from queue where id % {total}={ind}"):
            yield row(rec[0], self.decode_value(rec[1]))


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
