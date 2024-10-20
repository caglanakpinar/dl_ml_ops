import keras
import pandas as pd
from keras import layers, regularizers

from mlp import BaseData, BaseModel, Metrics, Params, log
from mlp.cli.cli import cli


class MyBinaryClassificationData(BaseData):
    """
    we will be using titanic csv file from
    https://storage.googleapis.com/tf-datasets/titanic/train.csv
    """

    def __init__(self, params: Params):
        self.params = params

    @classmethod
    def read(cls, params: Params, **kwargs):
        titanic_file = keras.utils.get_file("train.csv", params.get("data_url"))
        _cls = MyBinaryClassificationData(params)
        _cls.data = pd.read_csv(titanic_file)
        log(log.info, "One Hot Encoding for Categorical Features ...")
        for categorical_column in [
            "sex",
            "n_siblings_spouses",
            "parch",
            "class",
            "deck",
            "embark_town",
        ]:
            _dummies = pd.get_dummies(
                _cls.data[categorical_column], dtype=int, prefix=categorical_column
            )
            _cls.data = _cls.data.drop(categorical_column, axis=1)
            _cls.data = pd.concat([_cls.data, _dummies], axis=1)
        _cls.data["alone"] = _cls.data.alone.apply(lambda x: 0 if x == "n" else 1)
        return _cls


class MyBinaryClassificationModel(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        self.model = self.buildv1()

    def metrics(self):
        return [Metrics.train_epoch_metrics(metric) for metric in self.params.metrics]

    def buildv1(self) -> keras.Model:
        h_units = BaseModel.cal_hidden_layer_of_units(
            self.params.h_layers, self.params.units
        )
        _input = layers.Input(
            name=f"{self.params.model_type}_{self.params.name}_input",
            shape=(self.params.input_size,),
        )
        _hidden = layers.BatchNormalization()(_input)
        for _unit in h_units:
            _hidden = layers.Dense(
                _unit,
                activation=BaseModel.decision_of_activation(self.params.activation),
                use_bias=False,
            )(_hidden)
            _hidden = layers.Dropout(self.params.dropout)(_hidden)
        output = layers.Dense(
            self.params.output_size,
            name="output",
            activation=BaseModel.decision_of_activation(self.params.activation_output),
            use_bias=self.params.use_bias,
            kernel_regularizer=regularizers.l1_l2(l1=self.params.l1, l2=self.params.l2),
        )(_hidden)
        model = keras.Model(inputs=_input, outputs=output)
        model.compile(
            loss=self.params.loss,
            optimizer=self.optimizer(self.params.optimizer, self.params.lr),
            metrics=self.metrics(),
        )
        return model

    def train(self, dataset: BaseData.data_type):
        self.model.fit(
            # (True (train-val split), True (y variable available for this data), x (we are taking INPUT variables))
            x=dataset[(True, True, "x")],
            # (True (train-val split), True (y variable available for this data), x (we are taking TARGET variable))
            y=dataset[(True, True, "y")],
            # (True (train-val split), True (y variable available for this data),
            # validation_data (we are taking validation data tuple(x, y) ))
            validation_data=dataset[(True, True, "validation_data")],
            batch_size=self.params.batch_size,
            epochs=self.params.epochs,
        )


if __name__ == "__main__":
    cli()


# ---- BUILDING NEURAL NETWORK FROM A CONFIGURATION .yaml ----#
# from terminal, you can run for BINARY_CLASSIFICATION train
"""
poetry run python examples.py \
model train \
--data_access_class examples.MyBinaryClassificationData \
--trainer_config_path example_configurations/binary_classification_params.yaml \
--build_network_from_config True

"""
# from terminal, you can run for BINARY_CLASSIFICATION hyperparameter tuning
"""
poetry run python examples.py \
model tune \
--data_access_class examples.MyBinaryClassificationData
--trainer_config_path example_configurations/binary_classification_params.yaml
--build_network_from_config True
--hyper_parameter_config_path example_configurations/binary_classification_params_TUNE.yaml
"""

# --- BUILDING NEURAL NETWORK FROM A GIVEN TRAINER OBJECT ---#
"""
poetry run python examples.py \
model train \
--training_class examples.MyBinaryClassificationModel \
--data_access_class examples.MyBinaryClassificationData \
--trainer_config_path example_configurations/binary_classification_params_VALIDATION_SPIT.yaml \
--build_network_from_config False

"""