try:
    import panel as pn
except ImportError as e:
    raise ImportError('') from e
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
}
.bk {

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


long_text = HTMLTemplateFormatter(template=
                                  """
                     <details
                        style = "overflow: visible;" >
                     <summary > < % value['exception_type'] % > < / summary >
                     < p
                     style = 'display: inherit;white-space: break-spaces;word-wrap: normal;word-break: break-all;' >
                     < %-  exception_content % >
                     < / p >
                     < / details >
                     """)

tabulator_formatters = {
    'metrics': DictFormatter(column_name='metrics'),
    'progress': DictFormatter(column_name='progress'),
    'params': DictFormatter(column_name='params'),
    'exception': HTMLTemplateFormatter(template=ExceptionFormatter.base_template),
}

tabulator_editors = {
    'exp_name': None,
    'test_name': None,
    'params': None,
    'metrics': None,
    'progress': None,
    'exception': None,
    'str': {'type': 'list', 'valuesLookup': True},
    # 'note': TextEditor(),
    # 'tags': StringEditor(),
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

    # process progress
    ratio = df['progress'].apply(lambda x: x.get('ratio', 0))
    end_code = df['progress'].apply(lambda x: x.get('end_code', None))

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
            # 'rowHeight': 100,
            # 'columnDefaults': {
            #     'headerSort': False,
            #     },
        }
    )

    df_widget.on_edit(on_cell_change)
    return df_widget
