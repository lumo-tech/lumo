class ExpHook:

    def regist(self, exp): self.exp = exp

    def on_start(self, exp, *args, **kwargs): pass

    def on_end(self, exp, end_code=0, *args, **kwargs): pass

    def on_progress(self, exp, step, *args, **kwargs): pass

    def on_newpath(self, exp, *args, **kwargs): pass
