import datetime
from abc import abstractmethod

import keras
import tensorflow as tf
from keras import *
from keras.src.layers import LeakyReLU
from keras_tuner import HyperModel, HyperParameters, RandomSearch

from mlp import BaseData, log
from mlp.configs import Params
from mlp.utils import Paths


class BaseModel(Paths):
    @abstractmethod
    def __init__(self, params: Params):
        self.params = params
        self.continuous_training = self.params.continuous_training
        self.model: Model = Model()
        self.supervised = getattr(params, "target", None) is not None
        self.split = getattr(params, "split_ratio", None) is not None
        self.epoch_loss_metric = keras.metrics.Sum()
        NotImplementedError()

    def get_best_epoch_loss(self):
        return float(self.epoch_loss_metric.result().numpy())

    def checkpoint(self):
        if self.continuous_training:
            NotImplementedError()

    def checkpoint_model_default(self):
        if getattr(self.params, "checkpoint", None) is None:
            raise ValueError(
                """add <checkpoint_save_frequency> field to your configuration file. (integer)
                add <name> field which will create checkpoint folder such as 'training_checkpoints_<name>'
                add <checkpoint_monitor> field field to your configuration file.[OPTIONAL]. 
                Otherwise, default will be assigned as 'loss' 
                """
            )
        monitor = (
            self.params.get("checkpoint_monitor")
            if getattr(self.params, "checkpoint_monitor", None) is not None
            else "loss"
        )
        return keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_directory(self.params.get("name")),
            monitor=monitor,
            save_best_only=True,
            save_freq=self.params.get("checkpoint_save_frequency"),
        )

    @abstractmethod
    def train(self, dataset: tf.data.Dataset):
        NotImplementedError()

    def save(self, model: Model):
        file_name = (
            str(datetime.datetime.now())
            .replace("-", "")
            .replace(" ", "")
            .replace(":", "")[:12]
        )
        model.save(
            self.model_save_directory(self.params.get("name")) / f"{file_name}.keras"
        )

    def load(self, execution_time: str | None):
        files = os.listdir(self.model_save_directory(self.params.get("name")))
        model_saved_time = list(sorted([int(f.split(".")[0]) for f in files]))
        file_name = f"{max(model_saved_time)}.keras"
        if execution_time:
            execution_time = (
                execution_time.replace("-", "").replace(" ", "").replace(":", "")
            )
            exec_t_length = len(execution_time)
            if exec_t_length not in [12, 10]:
                raise log(
                    log.error,
                    """
                    execution date will only takes date or date with hour.
                    e.g. 2024-11-11, 20241111, 2024-11-11 10, 2024111110
                """,
                )
            if exec_t_length == 10:
                execution_time += "00"
            file_name = f"{execution_time}.keras"
            if int(execution_time) not in model_saved_time:
                date = list(
                    sorted(
                        [(abs(int(execution_time) - t), t) for t in model_saved_time],
                        key=lambda x: x[0],
                    )
                )[0][1]
                file_name = f"{date}.keras"
        return keras.models.load_model(
            self.model_save_directory(self.params.get("name")) / file_name
        )

    @classmethod
    def read_checkpoint(cls, params: Params):
        checkpoint_path = cls.checkpoint_directory(params.get("name"))
        if not checkpoint_path.exists():
            raise f"{checkpoint_path} - not found"
        model = cls(params)
        latest = tf.train.latest_checkpoint(checkpoint_path)
        model.checkpoint.restore(latest)
        return cls

    @staticmethod
    def cal_hidden_layer_of_units(
        hidden_layers, _encoding_dim, autoencoder_layers=False
    ):
        """creating hidden layers for each tower
        hidden_layers:
            number of hidden layer that will be created
        _encoding_dim:
            number of hidden unit that will be used in first hidden layer
        autoencoder_layers:
            if it is for autoencoder, process will not be same. hidden unit will be decreasing for each hidden layer,
            however, for autoencoder, after seeing bottle_neck unit, unit size will be re-increasing till _encoding_dim
        how it works;
            1st example_configurations;
                hidden_layers      : 3
                _encoding_dim      : 16
                autoencoder_layers : False
                layers             : 16 - 8 (16/2) - 4 (8/2) - 2 (4/2)
            2nd example_configurations;
                hidden_layers      : 3
                _encoding_dim      : 16
                autoencoder_layers : True
                layers             : 16 - 8 (16/2) - 4 (8/2) - 2 (4/2) (bottle_neck) - 4 (2*2) - 8 (4*2) - 16 (8*2)
        """
        count = 1
        _unit = _encoding_dim
        h_l_units = []
        while count != hidden_layers + 1:
            h_l_units.append(int(_unit))
            _unit /= 2
            if int(_unit) == 1:
                count = hidden_layers + 1
            else:
                count += 1
        if autoencoder_layers:
            count = 1
            while count != hidden_layers + 2:
                h_l_units.append(int(_unit))
                _unit *= 2
                count += 1
        return h_l_units

    @staticmethod
    def decision_of_activation(activation):
        """available activation functions for DL Network"""
        if activation == "sigmoid":
            return "sigmoid"
        if activation == "lrelu":
            return LeakyReLU(alpha=0.1)
        if activation == "relu":
            return "relu"
        if activation == "softmax":
            return "softmax"
        if activation == "tanh":
            return "tanh"

    @staticmethod
    def optimizer(opt, lr, **args):
        """decision for optimizer while applying hyper-parameter tuning"""
        if opt == "adam":
            return optimizers.Adam(learning_rate=lr)
        if opt == "rmsprop":
            return optimizers.RMSprop(learning_rate=lr)
        if opt == "sgd":
            return optimizers.SGD(learning_rate=lr, **args)
        if opt == "adadelta":
            return optimizers.Adadelta(learning_rate=lr, **args)
        if opt == "adagrad":
            return optimizers.Adagrad(learning_rate=lr, **args)
        if opt == "adamax":
            return optimizers.Adamax(learning_rate=lr, **args)


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

    def update_parameter(self, hp: HyperParameters):
        """updating parameters based on parameters coming from trial"""
        _selection_args = {
            p: (  # taking selected parameters from HyperParameters
                hp.Choice(p, getattr(self.temp_hyper_params, p))
                if type(getattr(self.temp_hyper_params, p)) == list
                else getattr(self.temp_hyper_params, p)
            )
            for p in self.temp_hyper_params.parameter_keys
        }
        _args = {  # updating parameter list. If parameters not exist on _selection_args use default from training_config
            p: (
                _selection_args.get(p)
                if p in _selection_args.keys()
                else getattr(self.temp_train_args, p)
            )
            for p in self.temp_train_args.parameter_keys
        }
        self.search_params = Params(
            trainer_arguments=_args
        )  # re-generating parameters by using params

    def random_search(self, model: HyperModel, dataset: BaseData.data_type, max_trials):
        """executing random trials by using Random Search algorith from keras_tuner
        !! this will update your trainer_config .yaml file by taking best parameters

        objective: always loss
        trials: available on trainer_config or hyperparameter_config
        directory: will create <self.tuning_project_dir> in the directory that is running for
        dataset: tuple, it will contain (x, y, validation_data)
        """
        tuner = RandomSearch(
            model,
            objective="loss",
            max_trials=max_trials,
            directory=self.parent_dir / self.tuning_project_dir,
        )
        tuner.search(
            x=dataset,
            y=dataset,
            validation_data=dataset,
        )
        self.temp_train_args.store_params(tuner.get_best_hyperparameters()[0].values)
