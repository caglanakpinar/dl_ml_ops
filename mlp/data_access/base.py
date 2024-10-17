from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from mlp.configs.configurations import Params
from mlp.utils import Paths


class BaseData(Paths):
    data: Any | tf.data.Dataset | np.ndarray = None
    train_dataset: Any | tf.data.Dataset | np.ndarray = None
    validation_dataset: Any | tf.data.Dataset | np.ndarray = None
    params: Params = None

    @abstractmethod
    def __init__(self, params: Params):
        self.params: Params = params
        NotImplementedError()

    @classmethod
    def read(cls, params: Params, **kwargs):
        NotImplementedError()

    def data_spliter(self, split_ratio: float):
        if type(self.data) not in [np.ndarray, pd.DataFrame]:
            data_size = len(self.data)
            self.data = self.data.shuffle(data_size)
            self.train_dataset = self.data.take(int(data_size * split_ratio))
            self.validation_dataset = self.data.skip(int(data_size * split_ratio))
        else:  # expecting only pandas dataframe
            self.train_dataset, self.validation_dataset = train_test_split(
                self.data, test_size=(1 - split_ratio), shuffle=True
            )
        if getattr(self.params, "target", None) is not None:
            self.train_dataset = {
                "x": self.train_dataset.drop(self.params.get("target"), axis=1),
                "y": self.train_dataset[[self.params.get("target")]],
            }
            self.validation_dataset = {
                "x": self.train_dataset.drop(self.params.get("target"), axis=1),
                "y": self.train_dataset[[self.params.get("target")]],
            }

    def get_data(self, split=False, split_ratio=0.8):
        assert getattr(self, "data", None) is not None, AttributeError(
            "pls make sure in data creation class has data attribute and all data set is available on"
        )
        if split:
            self.data_spliter(split_ratio)
            return self.train_dataset, self.validation_dataset
        else:
            return self.data
