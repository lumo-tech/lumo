import inspect
import os


def analyse_module_dependency(module, mem=None, root=None):
    """
    Recursively analyse the dependencies of a module and return a dictionary of modules and their associated files.

    Args:
        module (module): The module to analyse.
        mem (dict, optional): A dictionary of previously analysed modules and their associated files.
        root (str, optional): The root directory to use as a reference when determining whether a file is a dependency.

    Returns:
        dict: A dictionary of modules and their associated files.
    """
    if mem is None:
        mem = {}

    if inspect.ismodule(module):
        try:
            fn = getattr(module, '__file__', None)
        except RuntimeError:
            return mem
        if fn is None or fn in mem:
            return mem
        if os.path.commonprefix([os.path.abspath(fn), root]) != root:
            return mem
        mem[fn] = module
        for k, v in module.__dict__.items():
            analyse_module_dependency(v, mem, root)
    else:
        try:
            module = inspect.getmodule(module)
            if module is not None:
                analyse_module_dependency(module, mem, root)
        except:
            pass

    return mem
