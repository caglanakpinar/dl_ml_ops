from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from keras import Input, initializers, layers, regularizers
from keras.src.layers import BatchNormalization
from tensorflow import range

from mlp.configs import Params
from mlp.train.base import BaseModel


@dataclass
class TextLayers:
    """LSTM Text from Keras
    !!!lstm text layer is still under development. Currently, it is same as LSTMLayer!!
    batch_size:
        batch size will be sent to LSTM layer
    h_layers (integer) & units (integer):
        example_configurations:
            h_layers : 3
            units    : 16
            hidden layers will be like; 16 - 8 (16/2) - 4 (8/2)
    _input :
        input layer will be taken from model_data (from ModelData.x_train)
        input layer is filled in split task in execute.yaml in ml/split.py/TrainTestValidationSplit
    output :
        with 1 hidden unit
    return:
        _input :  _input
        _output: output with 1 dimension
        _contacted_layer: will be same as output layer which will be concatenated with other layers
    """

    params: Params

    def __call__(self) -> Tuple[Input, layers.Dense]:
        h_units = BaseModel.cal_hidden_layer_of_units(
            self.params.h_layers, self.params.units
        )
        _input = Input(
            name=f"{self.params.name}_input", shape=(self.params.input_size,)
        )
        _hidden = BatchNormalization()(_input)
        for _unit in h_units:
            _hidden = layers.LSTM(
                _unit,
                batch_size=self.params.batch_size,
                recurrent_initializer=initializers.Ones(),
                kernel_initializer=initializers.Ones(),
                use_bias=False,
                recurrent_activation=BaseModel.decision_of_activation(
                    self.params.activation
                ),
                dropout=self.params.dropout,
            )(_hidden)
        output = layers.Dense(1, name=self.params.name)(_hidden)
        return _input, output

    def token_position_embeddings(self, layer):
        token_emb = layer.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=self.params.embedding_dimensions,
        )
        pos_emb = layer.Embedding(
            input_dim=self.params.max_len, output_dim=self.params.embedding_dimensions
        )
        positions = range(start=0, limit=self.params.max_len, delta=1)
        positions = pos_emb(positions)
        x = token_emb(layer)
        return x + positions


@dataclass
class LSTMLayers:
    """LSTM from Keras
    batch_size:
        batch size will be sent to LSTM layer
    h_layers (integer) & units (integer):
        example_configurations:
            h_layers : 3
            units    : 16
            hidden layers will be like; 16 - 8 (16/2) - 4 (8/2)
    _input :
        input layer will be taken from model_data (from ModelData.x_train)
        input layer is filled in split task in execute.yaml in ml/split.py/TrainTestValidationSplit
    output :
        with 1 hidden unit
    return:
        _input :  _input
        _output: output with 1 dimension
        _contacted_layer: will be same as output layer which will be concatenated with other layers
    """

    params: Params

    def __call__(self) -> Tuple[Input, layers.Dense]:
        h_units = BaseModel.cal_hidden_layer_of_units(
            self.params.h_layers, self.params.units
        )
        _input = Input(
            name=f"{self.params.model_type}_{self.params.name}_input",
            shape=(self.params.lag, 1),
        )
        _hidden = BatchNormalization()(_input)
        for _unit in h_units:  # iteratively adding each layer with given units
            _hidden = layers.LSTM(
                _unit,
                recurrent_initializer=initializers.Ones(),
                kernel_initializer=initializers.Ones(),
                use_bias=False,
                recurrent_activation=BaseModel.decision_of_activation(
                    self.params.activation
                ),
                dropout=self.params.dropout,
            )(_hidden)
        output = layers.Dense(
            1, name=f"{self.params.model_type}_{self.params.name}_output"
        )(_hidden)
        return _input, output


@dataclass
class AutoEncoderLayers:
    params: Params

    def __call__(self) -> Tuple[Input, layers.Dense]:
        h_units = BaseModel.cal_hidden_layer_of_units(
            self.params.h_layers, self.params.units, autoencoder_layers=True
        )
        bottleneck_unit = min(h_units)
        _input = Input(
            name=f"{self.params.model_type}_{self.params.name}_input",
            shape=(self.params.input_size,),
        )
        _hidden = BatchNormalization()(_input)
        for _unit in h_units:
            _hidden = layers.Dense(
                _unit,
                activation=BaseModel.decision_of_activation(self.params.activation),
                use_bias=False,
                kernel_regularizer=regularizers.l1_l2(
                    l1=self.params.l1, l2=self.params.l2
                ),
            )(_hidden)
            if _unit == bottleneck_unit:
                _hidden._name = f"{self.params.model_type}_{self.params.name}_encoded"
            _hidden = layers.Dropout(self.params.dropout)(_hidden)
        output = layers.Dense(
            self.params.input_size,
            name=f"{self.params.model_type}_{self.params.name}_decoded",
            activation=BaseModel.decision_of_activation(self.params.activation_output),
            use_bias=False,
            kernel_regularizer=regularizers.l1_l2(l1=self.params.l1, l2=self.params.l2),
        )(_hidden)
        return _input, output


@dataclass
class MLPLayers:
    """Multi Layer Perceptron
    activation:
        activation function that is coming
        from hyper_parameter tuning yaml file tuned_params_.yaml which is available in docs folder
    use_bias:
        by default it is False, so, network will not create w0 weight matrix.
    h_layers (integer) & units (integer):
        example_configurations:
            h_layers : 3
            units    : 16
            hidden layers will be like; 16 - 8 (16/2) - 4 (8/2)
    _input :
        input layer will be taken from model_data (from ModelData.x_train)
        input layer is filled in split task in execute.yaml in ml/split.py/TrainTestValidationSplit
    output :
        with last hidden unit
    return:
        _input :  _input
        _output: last hidden unit
        _contacted_layer: will be same as output layer which will be concatenated with other layers
    """

    params: Params

    def __call__(self) -> Tuple[Input, layers.Dense]:
        h_units = BaseModel.cal_hidden_layer_of_units(
            self.params.h_layers, self.params.units
        )
        _input = Input(
            name=f"{self.params.model_type}_{self.params.name}_input",
            shape=(self.params.input_size,),
        )
        _hidden = BatchNormalization()(_input)
        for _unit in h_units:
            _hidden = layers.Dense(
                _unit,
                activation=BaseModel.decision_of_activation(self.params.activation),
                use_bias=False,
            )(_hidden)
            _hidden = layers.Dropout(self.params.dropout)(_hidden)
        return _input, _hidden


@dataclass
class OutputLayers:
    params: Params

    def __call__(self) -> Tuple[Input, layers.Dense]:
        h_units = BaseModel.cal_hidden_layer_of_units(
            self.params.h_layers, self.params.units
        )
        _input = Input(
            name=f"{self.params.model_type}_{self.params.name}_input",
            shape=(self.params.input_size,),
        )
        # _hidden = _input
        _hidden = BatchNormalization()(_input)
        for _unit in h_units:
            _hidden = layers.Dense(
                _unit,
                activation=BaseModel.decision_of_activation(self.params.activation),
                use_bias=False,
            )(_hidden)
            _hidden = layers.Dropout(self.params.dropout)(_hidden)
        output = layers.Dense(
            self.params.output_size,
            name=self.params.name,
            activation=BaseModel.decision_of_activation(self.params.activation_output),
            use_bias=False,
            kernel_regularizer=regularizers.l1_l2(l1=self.params.l1, l2=self.params.l2),
        )(_hidden)
        return _input, output
