from lumo.kit.params import ParamsType


class CallbackMix():
    def icallbacks(self, params: ParamsType):
        raise NotImplementedError()

    def add_callback(self, callback):
        raise NotImplementedError()

    def reload_callback(self, callback):
        raise NotImplementedError()

    def remove_callback(self, callback):
        raise NotImplementedError()


class ModelMix():

    def imodels(self, params: ParamsType):
        raise NotImplementedError()

    def ioptims(self, params: ParamsType):
        raise NotImplementedError()

    def optim_state_dict(self):
        raise NotImplementedError()

    def model_state_dict(self):
        raise NotImplementedError()
