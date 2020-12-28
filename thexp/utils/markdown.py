"""

"""
from contextlib import contextmanager


class Code():
    def __init__(self):
        self.lines = []

    def add_line(self, code):
        if isinstance(code, str):
            self.lines.append(code)
        elif isinstance(code, (list, tuple)):
            self.lines.extend(code)

    def new_line(self):
        self.lines.append("")

    def wrap(self):
        return "\n".join(self.lines)

    def __repr__(self):
        return self.wrap()


class Table():
    def __init__(self):
        self.lines = []
        self.heads = []
        self.mid = []
        self.col_len = None

    def update_col_len(self, line):
        if self.col_len is None:
            self.col_len = len(line)
        else:
            self.col_len = max(self.col_len, len(line))
            self.refresh_cols()

    def refresh_cols(self):
        if len(self.heads) != self.col_len:
            self.heads.extend(["-"] * (self.col_len - len(self.heads)))
        self.mid = [":---:"] * self.col_len
        for line in self.lines:
            if len(line) != self.col_len:
                line.extend(["-"] * (self.col_len - len(line)))

    def head(self, head: [list, tuple]):
        self.heads = head
        self.update_col_len(head)

    def append(self, line: [list, tuple]):
        self.lines.append(line)
        self.update_col_len(line)

    def mat(self):
        return [self.heads, *self.lines]

    def __repr__(self):
        res = []

        def add_line(line: list):
            res.append("|{}|\n".format("|".join([str(i) for i in line])))

        add_line(self.heads)
        add_line(self.mid)
        for line in self.lines:
            add_line(line)

        return "".join(res)


class Markdown():
    def __init__(self):
        self.lines = []

    def add_title(self, title, level=1):
        self.lines.append("\n{} {}\n".format("#" * level, title))
        return self

    def add_text(self, text):
        self.lines.append(text)
        return self

    def add_bold(self, text):
        self.lines.append(" **{}** ".format(text))
        return self

    def add_italic(self, text):
        self.lines.append(" *{}* ".format(text))
        return self

    def newline(self):
        # if len(self.lines) > 0 and len(self.lines[-1].strip()) != 0 and not self.lines[-1].endswith("\n"):
        self.lines.append("\n")
        return self

    def add_url(self, url, text):
        self.lines.append(" [{}]({}) ".format(text, url))
        return self

    def add_table(self, mat: list, head: list = None):
        with self.table() as tb:  # type:Table
            if head is None:
                tb.head(mat[0])
                mat = mat[1:]
            else:
                tb.head(head)

            for line in mat:
                tb.append(line)

    def __str__(self) -> str:
        return "".join(self.lines)

    def add_picture(self, url, text=None, inline=True):
        if inline:
            self.lines.append(" ![{}]({}) ".format(text, url))
        else:
            self.lines.append("\n\n![{}]({})\n\n".format(text, url))
        return self

    @contextmanager
    def table(self) -> Table:
        self.lines.append("\n\n")
        tb = Table()
        yield tb  # type:Table
        self.lines.append(str(tb))
        self.lines.append("\n\n")

    @contextmanager
    def parameter(self):
        self.lines.append("\n\n")
        yield self  # type:Markdown
        self.lines.append("\n\n")

    @contextmanager
    def quote(self):
        self.lines.append("\n\n")
        md = Markdown()
        yield md  # type:Markdown
        self.extends(md.wrap_quote())
        self.lines.append("\n\n")

    @contextmanager
    def code(self, lang=None) -> Code:
        """
        with md.code() as code:
            code.
        """
        self.lines.append("\n\n```{}\n".format(lang))
        code = Code()
        yield code
        self.lines.append(code.wrap())
        self.lines.append("\n```\n\n")

    def add_code(self, code, lang="", with_wrap=True):
        import textwrap
        if with_wrap:
            with self.code(lang) as cd:
                cd.add_line(textwrap.dedent(code).strip("\n"))
        else:
            self.lines.append("\n\n")
            self.lines.append(code)
            self.lines.append("\n\n")

        return self

    def extends(self, obj):
        if isinstance(obj, Markdown):
            self.lines.extend(obj.lines)
        elif isinstance(obj, (list, tuple)):
            self.lines.extend(obj)
        elif isinstance(obj, str):
            self.lines.append(obj)

        return self

    def wrap_quote(self):
        return ["> {}\n".format(i) for i in "".join(self.lines).split("\n")]

    def to_str(self):
        return str(self)

    def to_file(self, fn, mode="w"):
        with open(fn, mode, encoding="utf-8") as w:
            w.write(self.to_str())
