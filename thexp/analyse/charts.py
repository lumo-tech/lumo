"""
绘图类，用于和tensorboard配合
"""
from collections import defaultdict, OrderedDict

from itertools import chain, cycle

from typing import List, Dict, Union
import numpy as np
from numbers import Number


try:
    from pyecharts import charts
    from pyecharts import options as opts

except:
    import warnings
    warnings.warn("You need to install pyecharts to use charts.py, try pip install pyecharts.")

class Chart:
    def echarts(self):
        raise NotImplementedError

    def matplotlib(self):
        raise NotImplementedError


class Curve(Chart):

    def __init__(self, curve_values: Dict[str, Dict[str, Union[list, str]]], title="-", x_key='x', y_key='y',
                 name_key='name') -> None:
        super().__init__()
        self.title = title
        self.curve_values = curve_values
        self.x_key = x_key
        self.y_key = y_key
        self.name_key = name_key
        self._xaxis = None
        self._aligned = False

    @property
    def x_axis(self):
        if self._xaxis is None:
            self._xaxis = sorted(set(chain(*[v[self.x_key] for v in self.curve_values.values()])))
        return self._xaxis

    def _echarts_align_data(self):
        if self._aligned:
            return
        for k in self.curve_values.keys():
            v = self.curve_values[k]
            x_map = {x: y for x, y in zip(v[self.x_key], v[self.y_key])}
            new_ys = []
            for x in self.x_axis:
                if x in x_map:
                    new_ys.append(x_map[x])
                else:
                    new_ys.append(None)
            v[self.y_key] = new_ys

        self._aligned = True

    def echarts(self):
        self._echarts_align_data()
        c = charts.Line()

        for k, v in self.curve_values.items():
            c.add_xaxis([str(i) for i in self.x_axis])
            break

        # max_v_dict = defaultdict(list)

        for _, v in self.curve_values.items():
            # max_v_dict[k].append([max(v[self.y_key]),min(v[self.y_key])])
            c.add_yaxis(
                v[self.name_key],
                v[self.y_key],
                label_opts=opts.LabelOpts(is_show=False),
                is_symbol_show=False,
            )

        c.set_global_opts(title_opts=opts.TitleOpts(
            title=self.title, padding=5),
            yaxis_opts=opts.AxisOpts(
                min_='dataMin',
                max_='dataMax',
            ),
            legend_opts=opts.LegendOpts(type_='scroll', pos_bottom=10),
            tooltip_opts=opts.TooltipOpts(is_show=True, trigger='axis'), )

        return c

    def matplotlib(self):
        from matplotlib import pyplot as plt
        axes = plt.axes()

        names = []

        for v in sorted(self.curve_values.values(), key=lambda x: x[self.name_key]):
            # for _, v in self.curve_values.items():
            plt.plot(v[self.x_key], v[self.y_key])
            names.append(v[self.name_key])

        plt.legend(names)
        plt.title(self.title)
        return axes


class Parallel(Chart):
    def __init__(self, names: list, tag_dict: Dict[str, List[Number]]):
        super().__init__()
        self.names = names
        self.tag_dict = tag_dict

    def _echarts_axis_opts(self):
        """
        构建坐标系，按照值是string类或数字类，构建相应的坐标系
        Returns:

        """
        res = []
        for i, (k, v) in enumerate(self.tag_dict.items()):
            if isinstance(v[0], str):
                vv = list(set(v))
                vv.sort()
                res.append(opts.ParallelAxisOpts(dim=i, name=k, type_="category", data=vv))
            else:
                res.append(opts.ParallelAxisOpts(dim=i, name=k, max_=max(v), min_=min(v)))

        return res

    def _echarts_data(self):
        """
        返回 echarts需要的数据格式
        Returns:

        """
        res = []
        for i, test_name in enumerate(self.names):
            values = [values[i] for tag, values in self.tag_dict.items()]
            res.append([test_name, values])

        return res

    def echarts(self):
        c = charts.Parallel().add_schema(self._echarts_axis_opts())
        for name, data in self._echarts_data():
            c.add(name, data, is_smooth=True)

        return c

    def _matplotlib_data(self):
        test_key_vals = defaultdict(list)
        for (_, values) in self.tag_dict.items():
            for name, value in zip(self.names, values):
                test_key_vals[name].append(value)

        min_max = OrderedDict()
        for k, v in self.tag_dict.items():
            if isinstance(v[0], str):
                v_min, v_max = '', sorted(v)[0]
                dv = len(set(v)) + 1
            else:
                v_min, v_max = np.min(v), np.max(v)

                dv = v_max - v_min
                if dv == 0:
                    dv = abs(v_min)

                v_min -= dv * 0.05  # add 5% padding below and above
                v_max += dv * 0.05
                dv = v_max - v_min
            min_max[k] = [v_min, v_max, dv]

        return test_key_vals, min_max

    def matplotlib(self):
        from matplotlib import pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
        from itertools import cycle

        test_key_vals, min_max = self._matplotlib_data()
        key_size = len(min_max)
        host = plt.axes()
        # img, host = plt.subplots(figsize=(20,10))

        axes = [host] + [host.twinx() for _ in range(key_size - 1)]
        for i, ((k, v), ax) in enumerate(zip(min_max.items(), axes)):
            ax.set_ylim(v[0], v[1])
            if isinstance(v[0], str):
                ax.set_yticks([v[0], *sorted(list(set(self.tag_dict[k]))), '_'])

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if ax != host:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
                ax.spines["right"].set_position(("axes", i / (key_size - 1)))

        host.set_xlim(0, key_size - 1)
        host.set_xticks(range(key_size))
        host.set_xticklabels(list(self.tag_dict.keys()), fontsize=14)
        host.tick_params(axis='x', which='major', pad=7)
        host.spines['right'].set_visible(False)
        host.xaxis.tick_top()
        # host.set_title('Parallel Coordinates Plot', fontsize=18)

        colors = cycle(plt.cm.tab10.colors)
        for test, values in test_key_vals.items():
            # to just draw straight lines between the axes:
            # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            values = list(values)
            dv0 = None
            v_min0 = None
            for i, (k, (v_min, _, dv)) in enumerate(min_max.items()):
                if isinstance(v_min, str):
                    v_min = 0
                    values[i] = sorted(list(set(self.tag_dict[k]))).index(values[i]) + 1

                if i == 0:
                    dv0 = dv
                    v_min0 = v_min
                    continue
                values[i] = (values[i] - v_min) / dv * dv0 + v_min0

            n = 3
            verts = list(zip([x for x in np.linspace(0, key_size - 1, key_size * n - 2, endpoint=True)],
                             np.repeat(values, n)[1:-1]))
            # verts = list(zip(range(key_size), values))
            # verts = list(zip([0,1,1.5], [2.003,2.004,2.003]))
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=next(colors))
            host.add_patch(patch)

        host.legend(self.names, ncol=2, bbox_to_anchor=(0.75, -0.05))
        return host


class Table:
    pass


class Bar(Chart):
    pass
