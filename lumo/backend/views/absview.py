from typing import List


class View:
    """
    每个 view 可以嵌套其他的 view
    获取 info 的时候先获取本 view 的info，然后遍历 children 的 info
    children 的
    """

    def __init__(self, *, style=None, **children):
        self._parent = None
        self._children = children
        self._style = style

    def set_parent(self, parent):
        self._parent = parent

    def meta_info(self):
        return {}

    def children(self) -> List[str]:
        return sorted(self._children.keys())

    def get_children(self, name):
        return self._children.get(name, None)

    def render(self):
        res = {}
        res['info'] = self.meta_info()
        res['view'] = self.__class__.__name__
        children = res.setdefault('children', {})
        for k, v in self._children.items():
            if isinstance(v, View):
                children[k] = v.render()
            else:
                children[k] = v
        return res

    def __str__(self):
        return f'{self.__class__.__name__}({self.children()})'

    __repr__ = __str__
