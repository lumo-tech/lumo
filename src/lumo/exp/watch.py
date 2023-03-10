"""
Watcher 可以在运行实验后在 jupyter 或者网页上展示正在运行和已经运行结束的实验（按时间顺序？）
以及可以简化记录实验的烦恼

现在的核心痛点是
 - [ ] 所有元信息都有了，但是找不到哪个实验是哪个实验
 - [ ] 同时跑的多个实验有一个失败了，重跑时会混淆，或许需要一种覆盖手段 ->
 - > 怎么 rerun？
        lumo rerun test_name √
        lumo note html （用 streamlit 之类的生成动态网页）
        lumo note cmd  (类似 top 的视角，按时间顺序排列)
- > rerun 将已经跑的实验 move

可以代替 analysis 的作用。主要有

-> 按照 progress 目录，获取所有的实验
-> 根据获取的实验，按顺序记录
-> 每次只记录

"""
import os.path

from lumo.proc.path import progressroot

PID_ROOT = os.path.join(progressroot(), 'pid')
HB_ROOT = os.path.join(progressroot(), 'hb')
EXP_ROOT = os.path.join(progressroot())


# class Watcher:
#     """List and watch experiments with time order
#
#     Cache test_information in
#     metrics/<experiment>.sqlite
#     """
#
#     def load(self):
#         pass
#
#     def interactive(self):
#         """interactive, mark, label, note in ipython environment."""
#         pass
#
#     def server(self):
#         """simple server which make you note your experiments"""
#         pass
#
#     def list_all(self, exp_root=None, limit=100) -> Dict[str, List[Experiment]]:
#         """
#         Returns a dictionary of all experiments under exp_root directory.
#
#         Args:
#             exp_root: The root directory to search for experiments. Default is None, which uses the default experiment root directory.
#
#         Returns:
#             A dictionary of all experiments, where the keys are the names of the experiments and the values are lists of corresponding Experiment objects.
#         """
#         return {
#             _get_exp_name(exp_path): retrieval_tests_from_experiment(exp_path)
#             for exp_path in list_experiment_paths(exp_root)
#         }
