import keras
import numpy as np
import pandas as pd

from mlp import BaseData, BaseModel, Network, Params
from mlp.cli.cli import cli

reshape_3 = lambda x: x.reshape((x.shape[0], x.shape[1], 1))
reshape_2 = lambda x: x.reshape((x.shape[0], 1))


def drop_calculation(df, parameters, is_prediction=False):
    to_drop = max((parameters["tsteps"] - 1), (parameters["lahead"] - 1))
    df = df[to_drop:]
    if not is_prediction:
        to_drop = df.shape[0] % parameters["batch_size"]
        if to_drop > 0:
            df = df[: -1 * to_drop]
    return df


def data_preparation(df, f, parameters, is_prediction) -> dict:
    y = df[f].rolling(window=parameters["tsteps"], center=False).mean()
    x = pd.DataFrame(np.repeat(df[f].values, repeats=parameters["lag"], axis=1))
    shift_day = int(parameters["lahead"] / parameters["lag"])
    if parameters["lahead"] > 1:
        for i, c in enumerate(x.columns):
            x[c] = x[c].shift(i * shift_day)  # every each same days of shifted
    x = drop_calculation(x, parameters, is_prediction=is_prediction)
    y = drop_calculation(y, parameters, is_prediction=is_prediction)
    return split_data(y, x, parameters) if not is_prediction else reshape_3(x.values)


def split_data(Y, X, params):
    x_train = reshape_3(X[: -int(params["batch_count"] * params["batch_size"])].values)
    y_train = reshape_2(Y[: -int(params["batch_count"] * params["batch_size"])].values)
    x_test = reshape_3(X[-int(params["batch_count"] * params["batch_size"]) :].values)
    y_test = reshape_2(Y[-int(params["batch_count"] * params["batch_size"]) :].values)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


class ForecastData(BaseData):
    def __init__(self, params: Params):
        self.params = params

    @classmethod
    def read(cls, params: Params, **kwargs):
        _data = (
            pd.read_csv("./train.csv")
            .query("family == 'BEVERAGES' and store_nbr == 11")
            .sort_values(["date"])
        )
        _data.head()
        _cls = ForecastData(params)
        _cls.data = _data
        return _cls


class Forecast(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        self.norm = keras.layers.Normalization(axis=None)
        self.de_norm = keras.layers.Normalization(axis=None, invert=True)
        self.model = self.build()

    def get_time_series_params(self):
        return {
            "tsteps": self.params.get("tsteps"),
            "lahead": self.params.get("lahead"),
            "lag": self.params.get("lag"),
            "batch_size": self.params.get("batch_size"),
            "batch_count": self.params.get("batch_count"),
        }

    def build(self) -> keras.Model:
        network = Network(params=self.params)
        return network.model

    def train(self, dataset: BaseData.data_type):
        self.norm.adapt(dataset[self.params.get("lstm_target")].values)
        self.de_norm.adapt(dataset[self.params.get("lstm_target")].values)
        _dataset = data_preparation(
            dataset,
            [self.params.get("lstm_target")],
            self.get_time_series_params(),
            is_prediction=False,
        )
        self.model.fit(
            x=_dataset.get("x_train"),
            y=self.norm(_dataset.get("y_train")),
            validation_data=(_dataset.get("x_test"), self.norm(_dataset.get("y_test"))),
            batch_size=self.params.batch_size,
            epochs=self.params.epochs,
        )


if __name__ == "__main__":
    cli()

# --- BUILDING LSTM NEURAL NETWORK FROM A GIVEN TRAINER OBJECT ---#
"""
poetry run python examples_forecasting.py \
model train \
--training_class examples_forecasting.Forecast \
--data_access_class examples_forecasting.ForecastData \
--trainer_config_path example_configurations/forecast_params.yaml \
--build_network_from_config False
"""
