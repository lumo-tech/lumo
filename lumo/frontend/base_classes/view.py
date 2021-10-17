from lumo.utils.filebranch import FileBranch
from lumo.utils.safe_io import IO


def to_resource(data, code=200, msg=''):
    return {
        'data': data,
        'code': code,
        'msg': msg
    }


def auto_read(root: str):
    if root.endswith('log') or root.endswith('txt'):
        return {
            'flag': 'text',
            'content': IO.load_text(root)
        }
    elif root.endswith('json'):
        return {
            'flag': 'dict',
            'content': IO.load_json(root)
        }
    else:
        return {
            'flag': 'path',
            'content': root
        }


class View:
    """
    用来表示一个文件夹、一个文件的展示形式
    """
    flag = 'path'

    def __init__(self, root):
        self.fb = FileBranch(root)

    @property
    def root(self):
        return self.fb.root

    def jsonify(self):
        if isinstance(self.fb.isdir):
            return to_resource({
                'flag': self.flag,
                'content': self.fb.root
            })
        elif isinstance(self.fb.isfile):
            return to_resource(auto_read(self.fb.root))
        else:
            return to_resource({
                'flag': self.flag,
                'content': self.fb.root
            })


class LogView(View): pass


class ParamsView(View):
    flag = 'dict'


class TestView(View):
    def __init__(self, root):
        self.root = root

    def tags(self):
        pass
