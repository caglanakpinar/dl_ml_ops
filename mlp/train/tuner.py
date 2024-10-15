from abc import abstractmethod

from keras_tuner import HyperParameters, HyperModel, RandomSearch
import keras

from mlp.configs import Params
from mlp.data_access import Data


class BaseHyperModel(HyperModel):
    temp_model = None
    temp_train_args: Params = None
    temp_hyper_params: Params = None
    search_params: Params = None

    @staticmethod
    def get_hyper_parameters():
        return HyperParameters()

    def set_model(self, model):
        setattr(self, 'temp_model', model)

    def set_train_params(self, args):
        setattr(self, 'temp_train_args', args)

    def set_hyper_params(self, hyper_params):
        setattr(self, 'temp_hyper_params', hyper_params)

    @abstractmethod
    def build(self, hp: HyperParameters):
        NotImplementedError()

    @abstractmethod
    def fit(self, fp, model: keras.Model, **kwargs):
        NotImplementedError()

    @staticmethod
    def random_search(model: HyperModel, x, y, validation_data, max_trials):
        tuner = RandomSearch(
            model,
            objective='loss',
            max_trials=max_trials
        )
        tuner.search(x=x, y=x, validation_data=validation_data)

class Tuner:
    @staticmethod
    def tune(hyper_model: BaseHyperModel, trainer_config_path: str, hyper_parameter_config_path: str, data_builder: Data):
        params = Params(trainer_config_path)
        hyper_params = Params(hyper_parameter_config_path)
        model = hyper_model()
        model.set_model(model)
        model.set_train_params(params)
        model.set_hyper_params(hyper_params)
        model.random_search(
            model,
            x=data_builder.fetch_data(), y=data_builder.fetch_data(), validation_data=data_builder.fetch_data(),
            max_trials=hyper_params.get('max_trials')
        )
