from mlp.configs import Params
from mlp.data_access import Data
from mlp.train.base import BaseHyperModel


class Tuner:
    @staticmethod
    def tune(
        hyper_model: BaseHyperModel,
        trainer_config_path: str,
        hyper_parameter_config_path: str,
        data_builder: Data,
    ):
        params = Params(trainer_config_path)
        hyper_params = Params(hyper_parameter_config_path)
        model = hyper_model()
        model.set_model(model)
        model.set_train_params(params)
        model.set_hyper_params(hyper_params)
        model.random_search(
            model,
            x=data_builder.fetch_data(),
            y=data_builder.fetch_data(),
            validation_data=data_builder.fetch_data(),
            max_trials=hyper_params.get("max_trials"),
        )
