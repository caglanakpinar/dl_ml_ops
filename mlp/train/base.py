from abc import abstractmethod

import keras
import tensorflow as tf
from keras_tuner import HyperModel, HyperParameters, RandomSearch

from mlp.configs import Params
from mlp.utils import Paths


class BaseModel(Paths):
    @abstractmethod
    def __init__(self, params: Params):
        self.params = params
        NotImplementedError()

    @property
    def checkpoint(self):
        return tf.train.Checkpoint()

    @abstractmethod
    def train(self, train_dataset: tf.data.Dataset):
        NotImplementedError()

    @classmethod
    def read_checkpoint(cls, params: Params):
        checkpoint_path = cls.checkpoint_directory(params.get("name"))
        if not checkpoint_path.exists():
            raise f"{checkpoint_path} - not found"
        model = cls(params)
        latest = tf.train.latest_checkpoint(checkpoint_path)
        model.checkpoint.restore(latest)
        return cls


class BaseHyperModel(HyperModel, Paths):
    temp_model = None
    temp_train_args: Params = None
    temp_hyper_params: Params = None
    search_params: Params = None

    @staticmethod
    def get_hyper_parameters():
        return HyperParameters()

    def set_model(self, model):
        setattr(self, "temp_model", model)

    def set_train_params(self, args):
        setattr(self, "temp_train_args", args)

    def set_hyper_params(self, hyper_params):
        setattr(self, "temp_hyper_params", hyper_params)

    @abstractmethod
    def build(self, hp: HyperParameters):
        NotImplementedError()

    @abstractmethod
    def fit(self, fp, model: keras.Model, **kwargs):
        NotImplementedError()

    def random_search(self, model: HyperModel, x, y, validation_data, max_trials):
        tuner = RandomSearch(
            model,
            objective="loss",
            max_trials=max_trials,
            project_dir=self.parent_dir / self.tuning_project_dir,
        )
        tuner.search(x=x, y=y, validation_data=validation_data)
        self.temp_train_args.store_params(tuner.get_best_hyperparameters()[0].values)
