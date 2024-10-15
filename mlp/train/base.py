from abc import abstractmethod
import tensorflow as tf

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
        checkpoint_path = cls.checkpoint_directory(params.get('name'))
        if not checkpoint_path.exists():
            raise f"{checkpoint_path} - not found"
        model = cls(params)
        latest = tf.train.latest_checkpoint(checkpoint_path)
        model.checkpoint.restore(
            latest
        )
        return cls
