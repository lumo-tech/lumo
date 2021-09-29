#
# dataset = tf.data.experimental.SqlDataset("sqlite", "/foo/bar.sqlite3",
#                                           "SELECT name, age FROM people",
#                                           (tf.string, tf.int32))
# # Prints the rows of the result set of the above query.
# for element in dataset:
#   print(element)

from . import summary
from . import BatchIDSTrans
import sqlite3


class SqliteDataFrame:
    def __init__(self, db_file, mode='r'):
        self._conn = None
        self._cursor = None
        self.db_file = db_file
        self.mode = mode
        self.info = summary.summary_db(db_file)

    @property
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_file)
        return

    def writerow(self):
        pass

    def writerows(self):
        pass

    def sample(self, n):
        pass

    def head(self, n):
        pass

    def tail(self, n):
        pass

    def append(self, row):
        pass
