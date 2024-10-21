from keras import Model
from keras_tuner import HyperParameters

from mlp.configs import Params
from mlp.data_access import BaseData
from mlp.train.base import BaseHyperModel, BaseModel


class HyperNetwork(BaseHyperModel):
    def build(self, hp: HyperParameters):
        _selection_args = {
            p: (
                hp.Choice(p, getattr(self.temp_hyper_params, p))
                if type(getattr(self.temp_hyper_params, p)) == list
                else getattr(self.temp_hyper_params, p)
            )
            for p in self.temp_hyper_params.parameter_keys
        }
        _args = {
            p: (
                _selection_args.get(p)
                if p in _selection_args.keys()
                else getattr(self.temp_train_args, p)
            )
            for p in self.temp_train_args.parameter_keys
        }
        self.search_params = Params(trainer_arguments=_args)
        return Model()

    def fit(self, fp, model: Model, **kwargs):
        _model = self.temp_model(self.search_params)
        _model.split = True
        _model.train(dataset=kwargs["x"])
        return {"loss": _model.get_best_epoch_loss()}


class Tuner:
    @staticmethod
    def tune(
        hyper_model: BaseHyperModel,
        train_model: BaseModel,
        trainer_config_path: str,
        hyper_parameter_config_path: str,
        data_builder: BaseData,
    ):
        params = Params(trainer_config_path)
        hyper_params = Params(hyper_parameter_config_path)
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
