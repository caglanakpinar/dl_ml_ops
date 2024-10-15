from pathlib import Path


class Paths:
    checkpoint_dir = 'training_checkpoints'
    parent_dir = Path(__file__).absolute().parent

    def create_input_directory(self, name):
        file_path = self.parent_dir / Path(name)
        if not file_path.exists():
            Path.mkdir(self.parent_dir/ Path(name))
        return file_path

    @staticmethod
    def checkpoint_directory(name) -> Path:
        return Paths.parent_dir / f"{Paths.checkpoint_dir}_{name.upper()}"

    def create_train_epoch_image_save(self, name):
        folder_path = self.checkpoint_directory(name) / "ckpt"
        if not folder_path.exists():
            Path.mkdir(self.checkpoint_directory(name))
            Path.mkdir(folder_path)
        return folder_path