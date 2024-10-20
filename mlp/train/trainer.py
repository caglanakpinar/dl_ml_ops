from mlp import BaseData, Params, log
from mlp.train.base import BaseModel


class Trainer:
    def __init__(self, model: BaseModel, configurations: Params):
        self.model = model
        self.configurations = configurations

    @classmethod
    def create_trainer_from_config(cls, model: callable, configurations: Params):
        return Trainer(model=model(configurations), configurations=configurations)

    @classmethod
    def create_checkpoint_trainer_from_config(
        cls, model: callable, configurations: Params
    ):
        try:
            return Trainer(
                model=model.read_checkpoint(configurations),
                configurations=configurations,
            )
        except Exception as e:
            log(log.error, e)
            log(log.info, "checkpoint for the model is not available yet.")
            return Trainer(model=model(configurations), configurations=configurations)

    def trainer(self, data_class: BaseData):
        self.model.train(dataset=data_class.fetch_data())

    def get_model(self, attribute_name):
        assert getattr(self, attribute_name, None) is not None, AttributeError(
            f"model attribute name: {attribute_name} in training class"
        )
        return getattr(self, attribute_name)
