from abc import abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset

from mlp.configs.configurations import Params
from mlp.utils import Paths


class BaseData(Paths):
    default_data_type = Any | Dataset | np.ndarray | pd.DataFrame
    data_type = (
        default_data_type
        | Dict[
            tuple[str, str, str],
            default_data_type | tuple[default_data_type, default_data_type],
        ]
    )
    data: default_data_type = None
    train_dataset: data_type = None
    validation_dataset: data_type = None
    params: Params = None

    @abstractmethod
    def __init__(self, params: Params):
        self.params: Params = params
        NotImplementedError()

    @classmethod
    def read(cls, params: Params, **kwargs):
        NotImplementedError()

    def x(self, dataset: default_data_type) -> default_data_type:
        return dataset.drop(self.params.get("target"), axis=1)

    def y(self, dataset: default_data_type) -> default_data_type:
        return dataset[[self.params.get("target")]]

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

    def fetch_data(self, is_for_tuner=False) -> data_type:
        supervised = getattr(self.params, "target", None) is not None
        split = getattr(self.params, "split_ratio", None) is not None
        if is_for_tuner:
            split = supervised = True
            if getattr(self.params, "split_ratio", None) is None:
                setattr(self.params, "split_ratio", 0.8)
        assert getattr(self, "data", None) is not None, AttributeError(
            "pls make sure in data creation class has data attribute and all data set is available on"
        )
        if not split and not supervised:
            return self.data
        if not split and supervised:
            return {
                (split, supervised, "x"): self.x(self.data),
                (split, supervised, "y"): self.y(self.data),
                (split, supervised, "validation_data"): None,
            }
        if split:
            self.data_spliter(self.params.get("split_ratio"))
            if not supervised:
                return {
                    (split, supervised, "x"): self.train_dataset,
                    (split, supervised, "y"): self.train_dataset,
                    (split, supervised, "validation_data"): (
                        self.validation_dataset,
                        self.validation_dataset,
                    ),
                }
            else:  # supervised and split
                return {
                    (split, supervised, "x"): self.x(self.train_dataset),
                    (split, supervised, "y"): self.y(self.train_dataset),
                    (split, supervised, "validation_data"): (
                        self.x(self.validation_dataset),
                        self.y(self.validation_dataset),
                    ),
                }
