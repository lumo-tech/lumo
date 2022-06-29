import numpy as np
import streamlit as st
import sys

sys.path.insert(0, './src/lumo')
from . import parser

from lumo.exp import finder, Experiment
from datetime import date, timedelta


class HeadType:
    manully = 'Manully Type'
    query = 'Filter & Search'


class Main():
    def __init__(self):
        test_names = self.select_head()
        for test_name in test_names:
            test_root = finder.ensure_test_root(test_name)
            if test_root:
                # with st.expander(test_root):
                self.make_test(test_root)
                # st.write(test_root)
                # st.write(finder.format_experiment(Experiment.from_disk(test_root)))
            else:

                st.error(f'"{test_name}" is not a valid test name/ test root!')

    def make_test(self, test_root: str):
        exp = Experiment.from_disk(test_root)
        with st.expander("Experiment Info"):
            st.write(finder.format_experiment(exp))
        # with st.expander("Visualize Metrics"):
        if exp.has_prop('tensorboard_args'):
            tb = exp.get_prop('tensorboard_args')
            metrics = parser.parse_fron_tensorboard(tb['log_dir'])
        elif exp.has_prop('logger_args'):
            tb = exp.get_prop('logger_args')
            metrics = parser.parse_from_log(tb['log_dir'])
        else:
            metrics = {}
        metrics = list(metrics.items())
        for i in range(0, len(metrics), 2):
            l, m = st.columns(2)
            k, v = metrics[i]
            l.write(k)
            l.line_chart(np.array([vv.value for vv in v]))
            if i + 1 >= len(metrics):
                break
            k, v = metrics[i + 1]
            m.write(k)
            m.line_chart(np.array([vv.value for vv in v]))
                # if i + 2 >= len(metrics):
                #     break
                # k, v = metrics[i + 2]
                # r.line_chart({'k': np.array([vv.value for vv in v])})

    def select_head(self):
        left, right = st.columns([1, 3])
        left, right = st.sidebar, st.sidebar
        # left = st
        res = left.selectbox('', [HeadType.manully, HeadType.query])
        if res == HeadType.manully:
            test_names = self.manully_head(right)
        else:
            test_names = self.query_head(right)
        return test_names

    def manully_head(self, st):
        c = 0
        c = st.number_input('Test Number', c)
        test_names = []
        for i in range(c):
            test_names.append(st.text_input(f'Test ({i})', help='type test name or test root'))

        return test_names

    def query_head(self, st):
        sd = st
        self.experiments = finder.find_experiments()
        default_exp_name = None

        selection = sd.multiselect('Experiments', self.experiments, default=default_exp_name)

        test_names = []
        for select in selection:
            for test_name in finder.list_test_names_from_experiment(select):
                test_names.append(f'{select}/{test_name}')

        use_date_filter = False
        use_date_filter = sd.checkbox('Use Date Filter', use_date_filter)

        start = sd.date_input('Date Start Filter', date.today() - timedelta(days=7), disabled=not use_date_filter)
        end = sd.date_input('Date End Filter', date.today(), disabled=not use_date_filter)

        def unwrap(exp_test_name):
            return exp_test_name.split('/')[1]

        if use_date_filter:
            left = start.strftime('%y%m%d')
            right = end.strftime('%y%m%d')
            test_names = [i for i in test_names if left <= unwrap(i).split('.')[0] <= right]

        test_names = sd.multiselect('Tests', test_names)
        return [unwrap(i) for i in test_names]


def main():
    Main()


if __name__ == '__main__':
    main()
