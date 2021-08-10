import json
import sqlite3
import time
import random
import sys
import os

from lumo.base_classes import attr
from lumo.proc.path import cache_dir
from joblib import hash

class _NoneResult():
    def fetchone(self):
        return None


class MPStruct:
    TABLE_NAME = None
    TABLE_SQL = None
    INDEX_SQL = None

    def __init__(self, session_id=None, root=None, retry=7):
        if session_id is None:
            session_id = hash(sys.argv)

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
        for i in range(self._retry):
            try:
                res = self.cursor.execute(sql)
                if mode != 'r':
                    self.connect.commit()
                return res
            except sqlite3.OperationalError as e:
                from lumo.kit.logger import get_global_logger
                self.connect.rollback()
                get_global_logger().warn(f'[mpqueue] retry {i:02d}/{self._retry}...', sql, e)
                time.sleep(((i + 1) ** 2) * random.random())
                continue
        return _NoneResult()

    def executescript(self, sql, mode='r'):
        for i in range(self._retry):
            try:
                res = self.cursor.executescript(sql)
                if mode != 'r':
                    self.connect.commit()
                return res
            except sqlite3.OperationalError as e:
                from lumo.kit.logger import get_global_logger
                get_global_logger().warn(f'[mpqueue] retry {i:02d}/{self._retry}...', sql, e)
                # self.reconnect()
                time.sleep(random.random() * i + 0.5)
                continue
        return _NoneResult()
