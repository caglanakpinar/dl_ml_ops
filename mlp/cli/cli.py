from pydoc import locate

import click

from mlp.configs import Params
from mlp.data_access import BaseData, Data
from mlp.train import Trainer, Tuner
from mlp.train.tuner import BaseHyperModel


def import_class(path) -> object | BaseData | BaseHyperModel:
    if locate(path) is None:
        ImportError(f"model class not found in given path: {path}")
    else:
        return locate(path)


@click.group()
def cli():
    pass


@cli.group(name="model")
def model_run():
    pass


@model_run.command(
    name="train",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option(
    "--training_class",
)
@click.option(
    "--trainer_config_path",
)
@click.option(
    "--data_access_class",
)
@click.option("--continuous_training", default=False)
@click.option("--schedule", default=None)
def train(
    training_class,
    trainer_config_path,
    data_access_class,
    continuous_training,
    schedule,
    **kwargs,
):
    assert training_class is not None, AssertionError("training_class is missing")
    assert trainer_config_path is not None, AssertionError(
        "trainer_config_path is missing"
    )
    assert data_access_class is not None, AssertionError("data_access_class is missing")
    params = Params(trainer_config_path, **kwargs)
    if bool(continuous_training):
        trainer = Trainer.create_checkpoint_trainer_from_config(
            import_class(training_class), params
        )
    else:
        trainer = Trainer.create_trainer_from_config(
            import_class(training_class), params
        )
    databuilder = Data.create_trainer_from_config(
        import_class(data_access_class), params
    )
    trainer.trainer(databuilder.fetch_data(), **kwargs)


@model_run.command(
    name="tune",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option(
    "--tuning_class",
)
@click.option(
    "--trainer_config_path",
    default="configs/params.yaml",
    help="where trainer .yaml is being stored",
)
@click.option(
    "--hyper_parameter_config_path",
    default="configs/hyper_params.yaml",
    help="where hyperparameter tuning .yaml is being stored",
)
@click.option(
    "--data_access_class",
)
def tune(
    tuning_class,
    trainer_config_path,
    hyper_parameter_config_path,
    data_access_class,
    **kwargs,
):
    params = Params(trainer_config_path, **kwargs)
    databuilder = Data.create_trainer_from_config(
        import_class(data_access_class), params
    )
    Tuner.tune(
        import_class(tuning_class),
        trainer_config_path,
        hyper_parameter_config_path,
        databuilder,
    )
