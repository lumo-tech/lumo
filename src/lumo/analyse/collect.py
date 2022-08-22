from dbrecord import PDict
import pandas as pd


def collect(*database):
    res = []
    for exp in database:
        exp = PDict(exp)
        for v in exp.values():
            if isinstance(v, dict):
                res.append(v)
        exp.close()

    return pd.DataFrame(res)
