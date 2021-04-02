from lumo.utils.paths import hash



def cache():
    def wrap(func):
        def inner(*args,**kwargs):
            hashed_fn = hash([args,kwargs])
            func(*args,**kwargs)
