

def foo():
    a = 1
    b = 2
    raise NotImplementedError()

def fooo():
    c = 3
    foo()

try:
    fooo()
except:
    print(globals())
