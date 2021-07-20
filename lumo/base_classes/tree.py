"""

"""


class tree(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def walk(self):
        for k, v in self.items():
            yield k, v
            if isinstance(v, tree):
                for kk, vv in v.walk():
                    yield f'{k}/{kk}', vv
