import keras
import tensorflow as tf

from mlp.train.base import BaseModel


class Network(BaseModel):
    def __init__(self, **kwargs):
        self.model: keras.Model = None

    @classmethod
    def build(cls, **kwargs):
        network = Network(**kwargs)
        network.model = keras.Model()
        return network


    def train(self, train_dataset: tf.data.Dataset, **kwargs):
        self.model.fit(
            train_dataset,
            **kwargs
        )

class NeuralNetBuilder(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs

    @classmethod
    def builder(cls, **kwargs) -> BaseModel:
        builder = NeuralNetBuilder(**kwargs)
        builder.build(**kwargs)
        return builder
