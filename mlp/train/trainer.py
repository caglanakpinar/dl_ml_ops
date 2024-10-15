import tensorflow as tf

from mlp.configs.configurations import Params
from mlp.train.base import BaseModel
from mlp.train.builder import NeuralNetBuilder


class Trainer:
    def __init__(self, model: BaseModel, configurations: Params):
        self.model = model
        self.configurations = configurations

    @classmethod
    def create_trainer_from_config(cls, model:  callable, configurations: Params):
        return Trainer(
            model=model(
                configurations
            ),
            configurations=configurations
        )

    def trainer(self, train_dataset: tf.data.Dataset, build_by_config=False, **kwargs):
        if build_by_config:
            self.model = NeuralNetBuilder(**kwargs)
        self.model.train(train_dataset=train_dataset, **kwargs)

    def get_model(self, attribute_name):
        assert getattr(self, attribute_name, None) is not None, AttributeError(f"model attribute name: {attribute_name} in training class")
        return getattr(self, attribute_name)
