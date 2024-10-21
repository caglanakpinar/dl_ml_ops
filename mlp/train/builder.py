from typing import List, Tuple

from keras import *
from keras_tuner import HyperParameters

from mlp import BaseData, Metrics, Params, log
from mlp.train.base import BaseHyperModel, BaseModel
from mlp.train.models import (
    AutoEncoderLayers,
    LSTMLayers,
    MLPLayers,
    OutputLayers,
    TextLayers,
)


class Network(BaseModel):
    def __init__(self, params: Params):
        super().__init__(params)
        self.build_network_from_config()

    @property
    def metrics(self) -> list:
        return [Metrics.train_epoch_metrics(metric) for metric in self.params.metrics]

    @staticmethod
    def build_from_tower(params: Params) -> Tuple[Input, layers.Dense]:
        log(log.info, f"{params.model_type} - {params.name}")
        _call = AutoEncoderLayers(params)
        if params.model_type == "lstm":
            _call = LSTMLayers(params)
        if params.model_type == "text":
            _call = TextLayers(params)
        if params.model_type == "mlp":
            _call = MLPLayers(params)
        if params.model_type == "output":
            _call = OutputLayers(params)
        return _call()

    def final_layer(
        self, inputs: layers.Dense | List[layers.Dense], final_layer: layers.Dense
    ):
        output = layers.Dense(
            self.params.output_size,
            name="output",
            activation=BaseModel.decision_of_activation(self.params.activation_output),
            use_bias=self.params.use_bias,
            kernel_regularizer=regularizers.l1_l2(l1=self.params.l1, l2=self.params.l2),
        )(final_layer)
        return Model(inputs=inputs, outputs=output)

    def build_network_multi_towers_from_config(
        self,
        inputs: List[layers.Dense],
        towers: List[layers.Dense],
    ):
        concat_layers = layers.Concatenate()(towers)
        self.model = self.final_layer(inputs, concat_layers)

    def build_network_from_config(self):
        _input, _output = self.build_from_tower(self.params)
        self.model = self.final_layer(_input, _output)
        self.model.compile(
            loss=self.params.loss,
            optimizer=self.optimizer(self.params.optimizer, self.params.lr),
            metrics=self.metrics,
        )

    def train(self, dataset: BaseData.data_type, **kwargs):
        self.model.fit(
            x=dataset[(self.split, self.supervised, "x")],
            y=dataset[(self.split, self.supervised, "y")],
            validation_data=dataset[(self.split, self.supervised, "validation_data")],
            epochs=self.params.get("epochs"),
            callbacks=(
                None
                if not self.continuous_training
                else [self.checkpoint_model_default()]
            ),
        )


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
