import inspect
import os


def analyse_module_dependency(module, mem=None, root=None):
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
