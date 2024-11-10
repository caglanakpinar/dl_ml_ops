import os
from pathlib import Path


class Paths:
    checkpoint_dir = "training_checkpoints"
    checkpoint_prefix = "ckpt"
    parent_dir = Path(os.getcwd())
    tuning_project_dir = "hyper_parameter_tuning"

    def create_image_directory(self, name):
        file_path = self.parent_dir / Path(name)
        if not file_path.exists():
            Path.mkdir(self.parent_dir / Path(name))
        return file_path

    @staticmethod
    def checkpoint_directory(name) -> Path:
        return Paths.parent_dir / f"{Paths.checkpoint_dir}_{name.upper()}"

    @staticmethod
    def model_save_directory(name) -> Path:
        return Paths.parent_dir / name.upper()

    def checkpoint_prefix_directory(self, name):
        return self.checkpoint_directory(name) / self.checkpoint_prefix

    def create_directory_in_parents(self, dir):
        folder_path = self.parent_dir / dir
        if not folder_path.exists():
            Path.mkdir(folder_path)
        return folder_path
