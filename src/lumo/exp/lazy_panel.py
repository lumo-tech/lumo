try:
    import panel as pn
except ImportError as e:
    raise ImportError('The experiment panel is supported by panel, '
                      'you should use it by `pip install panel` first.') from e

from typing import Any

from bokeh.core.property.primitive import String
from bokeh.plotting import figure
import pandas as pd
from panel.models.tabulator import TableEditEvent

from lumo.exp.watch import Watcher
import datetime as dt
import numpy as np
from bokeh.models.widgets.tables import HTMLTemplateFormatter, NumberFormatter, TextEditor, StringEditor
from lumo import Experiment

css = '''
.tabulator-tableholder {
 height: fit-content !important;
}

.tabulator-row .tabulator-cell {
  overflow: visible !important;
  vertical-align: top;
  min-height: 20px;
  height: fit-content !important;
}

.bk {
 height: fit-content !important;
}

.tabulator .tabulator-col-resize-handle {
 height: fit-content !important;

}

.tabulator-cell .tabulator-editable {
 width: fit-content !important;
 
}

'''

pn.extension('tabulator', raw_css=[css], css_files=[pn.io.resources.CSS_URLS['font-awesome']])


def DictFormatter(column_name):
    base_template = """
    <details style="overflow: visible;">
        <summary>{column_name}</summary>
        <% _.each(value, function(vv, key) { %>
            <li><b><%= key %>:</b> <%= vv %></li>
        <% }); %>
    </details>"""
    return HTMLTemplateFormatter(template=base_template.replace('{column_name}', column_name))


class ExceptionFormatter:
    base_template = """
<details style="overflow: visible;">
    <summary> 
        <%= value['exception_type'] %> 
    </summary>
    <p style='display: inherit;white-space: break-spaces;word-wrap: normal;word-break: break-all;'> 
        <%= value['exception_content'] %> 
    </p>
</details>"""


progress_formatter = HTMLTemplateFormatter(
    template="""
    <div style="min-height: 1.5em; width: <%-  value.ratio %>; background-color: <%-  value.color %>;" ></div>
    """
)

tabulator_formatters = {
    'metrics': DictFormatter(column_name='metrics'),
    'progress_': DictFormatter(column_name='progress'),
    'progress': progress_formatter,
    'params': DictFormatter(column_name='params'),
    'exception': HTMLTemplateFormatter(template=ExceptionFormatter.base_template),
}

tabulator_editors = {
    'exp_name': None,
    'test_name': None,
    'params': None,
    'metrics': None,
    'progress': None,
    'progress_': None,
    'exception': None,
    'str': {'type': 'list', 'valuesLookup': True},
}


def make_experiment_tabular(df: pd.DataFrame, reload_fn):
    """
    {'start': '23-03-14-224651',
     'finished': False,
     'ratio': 0,
     'last_edit_time': '23-03-14-224651',
     'update_from': None}

    Args:
        df:
        reload_fn:

    Returns:

    """
    # process exception
    if 'exception' not in df.columns:
        df['exception'] = {}
    else:
        df['exception'].fillna({})
    if 'tags' not in df.columns:
        df['tags'] = np.empty((len(df.index), 0)).tolist()

    def reformat_progress(dic):
        if not isinstance(dic, dict):
            return {
                'ratio': '100%',
                'color': 'yellow',
            }

        ratio = dic.get('ratio', 0)
        end_code = dic.get('end_code', None)

        # normal end ->  end_code == 0 -> green
        # interrupt by exception ->  end_code > 0 -> red
        # running -> end_code is None -> blue
        # killed without any signal ( in 5 minute ) ->  end_code is None -> blue
        # killed and dumped by watcher ->  end_code == -10 -> red

        if end_code is None:
            color = 'blue'
        elif ratio == 1:
            color = 'green'
        elif end_code == 0:
            color = 'green'
        elif end_code > 0:
            color = 'red'
            if ratio == 0:
                ratio = 0.01

        return {
            'ratio': f'{ratio:2%}',
            'color': color,
        }

    df['progress_'] = df['progress']
    df['progress'] = df['progress'].apply(reformat_progress)

    # ratio = 1
    # ratio < 1, no-end-code ->

    df = df[[
        'exp_name', 'test_name',
        'progress',
        'exception',
        'params', 'metrics',
        'note', 'tags',
    ]]

    def on_cell_change(e: TableEditEvent):
        nonlocal df

        if e.column == 'note':
            Experiment.from_cache(df.iloc[e.row].to_dict()).dump_note(e.value)
        else:
            tags = [i.strip() for i in e.value.split(',')]
            Experiment.from_cache(df.iloc[e.row].to_dict()).dump_tags(tags)

        df = reload_fn()

    df_widget = pn.widgets.Tabulator(
        df,
        groupby=['exp_name'],
        hidden_columns=['git', 'paths', 'pinfo',
                        'execute', 'lock', 'params.yaml',
                        'params_hash', 'table_row', 'hooks', 'logger_args',
                        'metric_board'],
        pagination='local',
        formatters=tabulator_formatters,
        editors=tabulator_editors,
        selection=[],
        show_index=False,
        configuration={
            'clipboard': True,
            'tooltip': True,
            # 'rowHeight': 100,
            # 'columnDefaults': {
            #     'headerSort': False,
            #     },
        }
    )

    # def on_cell_click(e):
    #     nonlocal enable_update
    #     if e.column in {'note', 'tags'}:
    #         enable_update = False

    # enable_update = True
    # df_widget.on_click(on_cell_click)
    df_widget.on_edit(on_cell_change)
    #
    # def update_time():
    #     if enable_update:
    #         time_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         time_component.object = f"<p>{time_now}</p>"
    #
    # time_component = pn.pane.HTML("current time is")
    # pn.state.onload(update_time)
    # pn.state.add_periodic_callback(update_time, 1000)  # 每1000毫秒执行一次更新时间的操作
    #
    # widget = pn.Column(time_component, df_widget)  # .servable()

    return df_widget
