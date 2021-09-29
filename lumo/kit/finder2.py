"""
Help you find the experiments you have done.
"""
from .experiment import Experiment
from lumo.proc.path import libhome
from lumo.utils.filebranch import FileBranch

#
# print(list(FileBranch(libhome()).branch('experiment').find_dir_in_depth('.*', 0)))
# print(list(FileBranch(libhome()).branch('experiment').find_dir_in_depth('[0-9.a-z]{13}t$', 1)))
