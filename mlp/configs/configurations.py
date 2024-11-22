from pathlib import Path

import yaml

from mlp.utils import Paths


class Params(Paths):
    """default parameters for Network"""

    name: str = ""
    input_size: int | tuple = 10
    output_size: int = 1
    activation: str = "relu"
    activation_output: str = "sigmoid"
    use_bias: bool = False
    batch_size: int = 64
    epochs: int = 200
    units: int = 16
    h_layers: int = 3
    dropout: float = 0.0
    l1: float = 0.0001
    l2: float = 0.0001
    lr: float = 0.0001
    loss: str = "binary_crossentropy"
    optimizer: str = "adam"
    model_type: str = "mlp"
    # below parameters will be used only for LSTM-Text layers
    vocab_size: int = 1_000
    embedding_dimensions: int = 5
    max_len: int = 10
    continuous_training: bool = False
    # lstm parameters
    tsteps: int = 1
    lahead: int = 1
    lag: int = 7

    def __init__(
        self,
        trainer_config_path: Path | str = None,
        trainer_arguments: dict = None,
        **kwargs,
    ):
        self.parameter_keys = []
        self.trainer_config_path = trainer_config_path
        self.read_from_config(trainer_config_path, trainer_arguments, **kwargs)

    def get(self, p):
        assert (
            getattr(self, p, None) is not None
        ), f"<{p}> - is not available at train parameters .yaml file"
        return getattr(self, p)

    def read_from_config(
        self, trainer_config_path, trainer_arguments: dict = None, **kwargs
    ):
        if trainer_arguments is None:
            trainer_arguments = self.read_yaml(self.parent_dir / trainer_config_path)
        setattr(self, "parameter_keys", [*trainer_arguments.keys()])
        for p, value in trainer_arguments.items():
            setattr(self, p, value)
        if kwargs is not None:
            for p, value in kwargs.items():
                setattr(self, p, value)

    def store_params(self, params: dict):
        updated_params = {}
        for p, value in self.read_yaml(
            self.parent_dir / self.trainer_config_path
        ).items():
            updated_params[p] = params.get(p, value)
        self.write_yaml(self.parent_dir / self.trainer_config_path, updated_params)

    @staticmethod
    def read_yaml(folder):
        """
        :param folder: file path ending with .yaml format
        :return: dictionary
        """
        with open(
            f"{str(folder)}.yaml"
            if str(folder).split(".")[-1] not in ["yaml", "yml"]
            else folder
        ) as file:
            docs = yaml.full_load(file)
        return docs

    @staticmethod
    def write_yaml(folder, params: dict):
        """
        :param folder: file path ending with .yaml format
        :param params: dict to .yaml format
        """
        with open(
            (
                f"{str(folder)}.yaml"
                if str(folder).split(".")[-1] not in ["yaml", "yml"]
                else folder
            ),
            "w",
        ) as file:
            yaml.dump(params, file)
