from pathlib import Path

import yaml

from mlp.utils.paths import Paths


class Params(Paths):
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
        ), f"{p} - is not available at train parameters .yaml file"
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
            f"{str(folder)}.yaml"
            if str(folder).split(".")[-1] not in ["yaml", "yml"]
            else folder,
            "w",
        ) as file:
            yaml.dump(params, file)
