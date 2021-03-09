"""

"""
import os

from ..base_classes.defaults import draw_dict
from ..utils.dates import curent_date
from ..decorators import deprecated


@deprecated('1.4.0', '1.5.0')
class Reporter():
    """
    记录并生成 markdown 格式的报告
    """

    def __init__(self, pltf_dir, exts=None):

        self.base_dir = pltf_dir
        from collections import defaultdict
        self.cur_key = curent_date()
        self.plot_dict = defaultdict(draw_dict)
        self.cur_dir = None
        self.ftags = []
        self.tags = set()

    @property
    def savedir(self):
        if self.cur_dir is None:
            i = 1
            fn = os.path.join(self.base_dir, '{:04}'.format(i))
            while os.path.exists(fn):
                i += 1
                fn = os.path.join(self.base_dir, '{:04}'.format(i))
            os.makedirs(fn, exist_ok=True)
            self.cur_dir = fn

        return self.cur_dir

    def as_numpy(self, var):
        import torch, numpy as np
        if isinstance(var, (int, float)):
            return var

        if isinstance(var, torch.Tensor):
            return var.detach().cpu().item()

        if isinstance(var, np.ndarray):
            return var.item()

        assert False

    def add_scalar(self, var, step, tag):
        self.plot_dict[tag]['x'].append(step)
        self.plot_dict[tag]['y'].append(self.as_numpy(var))

    def add_ftag(self, file, tag):

        self.ftags.append((os.path.normpath(file), tag))

    def savefig(self):
        from matplotlib import pyplot as plt
        from thexp.utils.paths import filter_filename
        dir = self.savedir
        for k, v in self.plot_dict.items():
            if 'fn' in v:
                continue

            plt.figure()
            base_name = filter_filename('{}.jpg'.format(k))
            fn = os.path.join(dir, base_name)
            plt.plot(v['x'], v['y'])
            plt.title(k)
            plt.savefig(fn)
            v['fn'] = base_name
        return dir

    @property
    def picklefile(self):
        return 'analyser.pkl'

    @property
    def reportfile(self):
        return 'report.md'

    def add_tag(self, tag):
        self.tags.add(str(tag))

    def report(self, every=20, otherinfo=None):
        every = max(every, 1)

        from thexp.utils.markdown import Markdown
        md = Markdown()
        md.add_title(curent_date('%y-%m-%d-%H:%M:%S'))
        with md.code() as code:
            for tag in self.tags:
                code.add_line(tag)

        if otherinfo is not None:
            md.extends(otherinfo)
        md.add_title('Vars', level=2)
        self.savefig()
        for k, v in self.plot_dict.items():
            md.add_title(k, level=3)
            if 'fn' in v:
                md.add_picture('./{}'.format(v['fn']), k, False)

            lines = []
            head = ['step']
            vars = ['values']

            xs = []
            ys = []

            for x, y in zip(v['x'][::every], v['y'][::every]):
                head.append('**{}**'.format(x))
                vars.append('{:.4f}'.format(y))
                xs.append(x)
                ys.append(y)
                if len(head) > 10:
                    lines.append(head)
                    lines.append(vars)
                    head = ['**step**']
                    vars = ["values"]

            if len(head) != 1:
                lines.append(head)
                lines.append(vars)

            ys = [float("{:.4f}".format(y)) for y in v["y"]]
            md.add_table(lines)
            code = """
            every = {}
            xs = {}
            ys = {}
            plt.plot(xs[::every],ys[::every])
            plt.title({})
           """.format(every, v["x"], ys, k)
            md.add_code(code, "python")

        md.add_title("Files", level=2)
        with md.quote() as q:
            q.add_text("PWD : {}".format(os.getcwd()))
        with md.table() as table:
            table.head(["index", "tag", "file"])
            for i, (file, tag) in enumerate(self.ftags):
                table.append([i, tag, file])

        dir = self.savedir
        fn = os.path.join(dir, self.reportfile)
        md.to_file(fn)
        return fn
