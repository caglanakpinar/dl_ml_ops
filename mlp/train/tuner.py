from mlp.configs import Params
from mlp.data_access import BaseData
from mlp.train.base import BaseHyperModel, BaseModel
from mlp.train.builder import Network


class Tuner:
    @staticmethod
    def tune(
        hyper_model: BaseHyperModel,
        train_model: BaseModel,
        trainer_config_path: str,
        hyper_parameter_config_path: str,
        data_builder: BaseData,
        build_network_from_config: bool,
    ):
        params = Params(trainer_config_path)
        hyper_params = Params(hyper_parameter_config_path)
        if build_network_from_config:
            train_model = Network
        model = hyper_model()
        model.set_model(train_model)
        model.set_train_params(params)
        model.set_hyper_params(hyper_params)
        _dataset = data_builder.fetch_data(is_for_tuner=True)
        model.random_search(
            model,
            dataset=_dataset,
            max_trials=hyper_params.get("max_trials"),
        )
