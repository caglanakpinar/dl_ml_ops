from abc import abstractmethod

from mlp.configs.configurations import Params


class BaseData:
    @abstractmethod
    def __init__(self, params: Params):
        self.params = params
        NotImplementedError()

    @abstractmethod
    def read(self, **kwargs):
        NotImplementedError()

    def get_data(self):
        assert getattr(self, 'data', None) is not None, AttributeError(f"pls make sure in data creation class has data attribute and all data set is available on")
        return getattr(self, 'data')