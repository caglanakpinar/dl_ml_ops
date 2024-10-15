from mlp.configs.configurations import Params
from mlp.data_access.base import BaseData


class Data:
    def __init__(self, dataset_creator, configurations: Params):
        self.dataset_creator = dataset_creator
        self.configurations = configurations


    @classmethod
    def create_trainer_from_config(cls, dataset_creator:  BaseData, configurations: Params):
        return Data(
            dataset_creator=dataset_creator(
                configurations
            ),
            configurations=configurations
        )

    def fetch_data(self, **kwargs) -> object:
        _cls: BaseData = self.dataset_creator.read(self.configurations)
        return _cls.get_data()