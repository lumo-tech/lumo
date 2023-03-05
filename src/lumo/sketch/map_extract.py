def dicmap(kout=None, **maps):
    def wrap(function):
        def inner(mem: dict):
            fkwargs = {k: mem[v] for k, v in maps.items()}
            res = function(**fkwargs)
            if isinstance(res, dict):
                for k in res:
                    mem[k] = res[k]
            elif isinstance(res, (list, tuple)) and isinstance(kout, (list, tuple)):
                assert len(res) == len(kout)
                for k, v in zip(kout, res):
                    mem[k] = v
            elif isinstance(kout, str):
                mem[kout] = res

        return inner

    return wrap
