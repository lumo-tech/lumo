from lumo.nest.params import DataLoaderPM,BaseParams
from lumo import Params

class My(Params,DataLoaderPM):
    pass

a = My()

print(a)