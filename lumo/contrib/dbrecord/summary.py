import pprint
import sqlite3
import os
from typing import Union, List


def summary_table_struct(cols):
    """
    c = conn.execute("PRAGMA table_info(GEN_01);")
    desc = c.description
    cols = c.fetchall()

    summary_table_struct(desc,cols)

    Args:
        desc:
        cols:

    Returns:

    """
    # desc = [i[0] for i in desc]
    info = {}
    for cid, name, typ, notnull, default, pk in cols:
        info[name] = {}
        info[name]['cid'] = cid
        info[name]['type'] = typ
        info[name]['notnull'] = notnull
        info[name]['default'] = default
        info[name]['pk'] = pk

    return info


def summary_db(sqlite_file, columns=True,
               row_count=True,
               struct_info=True):
    info = {}
    if os.path.exists(sqlite_file):
        info['exists'] = 'ok'
    else:
        info['exists'] = 'not exists'
        return info

    try:
        conn = sqlite3.connect(sqlite_file)
    except sqlite3.DatabaseError:
        info['is_database'] = False
        return info

    tables = conn.execute("select name from sqlite_master where type='table';").fetchall()
    tables = [i[0] for i in tables if not i[0].startswith('sqlite')]
    info['db_count'] = len(tables)
    info['db_names'] = tables

    if row_count:
        for table_name in tables:
            row_count = count_table(conn, table_name)
            res = info.setdefault(table_name, {})
            res['row_count'] = row_count

    if columns:
        for table_name in tables:
            c = conn.execute(f"PRAGMA table_info({table_name});")
            # desc = c.description
            res = info.setdefault(table_name, {})
            res['columns'] = summary_table_struct(cols=c.fetchall())

    if struct_info:
        return info
    else:
        return pprint.pformat(info)


def check_db_table_ok(db: str, table: str = None, cols: Union[List[str], str] = None):
    info = summary_db(db, columns=True, row_count=False, struct_info=True)

    if info['exists'] != 'ok':
        return False, f'File {db} not exists.'

    if not info['is_database']:
        return False, f'File {db} is not a database.'

    if table is not None:
        if table not in info['db_names']:
            return False, f'Table {table} not in database({db})'

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]

            real_cols = {i.lower() for i in info[table]['columns'].keys()}
            for col in cols:
                if col not in real_cols:
                    return False, f'Column {col} not in table {table}({real_cols}).'
    return True, 'success'


def count_table(conn, table):
    """
    Calculate table row count. More quick than executing `count(*)`.

    %timeit count_table(conn,table_name)
    276 µs ± 936 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    %timeit conn.execute(f'select count(*) from {table_name}').fetchone()
    43.6 ms ± 186 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    Warnings:
        通过二分法计数，不能有任何行被删除，即所有 row id 必须完整，且有索引
    """
    left = 0
    idx = 10000
    right = None

    while right is None:
        res = conn.execute(f'select * from {table} where id == {idx}').fetchone()
        if res is not None:
            idx *= 2
        else:
            right = idx

    right = idx

    while right - left != 1:
        res = conn.execute(f'select * from {table} where id == {idx}').fetchone()
        if res is not None:
            left = idx
            nxt = (idx + right) // 2
            idx = nxt
        else:
            right = idx
            nxt = (left + right) // 2
            idx = nxt
    return idx


def count_table2(conn, table):
    return conn.execute(f"select count(*) from {table};").fetchone()[0]
