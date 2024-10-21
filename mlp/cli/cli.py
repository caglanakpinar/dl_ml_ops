from pydoc import locate

import click

from mlp import BaseHyperModel, BaseModel, HyperNetwork, Network, Trainer, Tuner
from mlp.configs import Params
from mlp.data_access import BaseData


def import_class(path) -> object | BaseData | BaseModel | BaseHyperModel:
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
@click.option("--training_class", required=False)
@click.option(
    "--trainer_config_path",
    default="configs/params.yaml",
    help="where trainer .yaml is being stored",
)
@click.option("--data_access_class", required=True)
@click.option("--continuous_training", default=False)
@click.option("--build_network_from_config", default=False)
@click.option("--schedule", default=None)
def train(
    training_class,
    trainer_config_path,
    data_access_class,
    continuous_training,
    build_network_from_config,
    schedule,
    **kwargs,
):
    if build_network_from_config:
        assert training_class is not None, AssertionError("training_class is missing")
    assert trainer_config_path is not None, AssertionError(
        "trainer_config_path is missing"
    )
    assert data_access_class is not None, AssertionError("data_access_class is missing")
    params = Params(
        trainer_config_path,
        **{
            **kwargs,
            **{
                "continuous_training": continuous_training,
                "build_network_from_config": build_network_from_config,
            },
        },
    )
    training_class: BaseModel = (
        Network if build_network_from_config else import_class(training_class)
    )
    databuilder: BaseData = import_class(data_access_class).read(params)
    if bool(continuous_training):
        trainer = Trainer.create_checkpoint_trainer_from_config(training_class, params)
    else:
        trainer = Trainer.create_trainer_from_config(training_class, params)
    trainer.trainer(databuilder, **kwargs)


@model_run.command(
    name="tune",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--tuning_class", required=False)
@click.option("--training_class", required=False)
@click.option(
    "--trainer_config_path",
    default="configs/params.yaml",
    help="where trainer .yaml is being stored",
)
@click.option(
    "--hyperparameter_config_path",
    default="configs/hyper_params.yaml",
    help="where hyperparameter tuning .yaml is being stored",
)
@click.option("--data_access_class", required=True)
@click.option("--build_network_from_config", default=False)
def tune(
    tuning_class,
    training_class,
    trainer_config_path,
    hyperparameter_config_path,
    data_access_class,
    build_network_from_config,
    **kwargs,
):
    if build_network_from_config:
        assert training_class is not None, AssertionError("training_class is missing")
        assert tuning_class is not None, AssertionError("tuning_class is missing")
    assert trainer_config_path is not None, AssertionError(
        "trainer_config_path is missing"
    )
    assert hyperparameter_config_path is not None, AssertionError(
        "hyperparameter_config_path is missing"
    )
    assert data_access_class is not None, AssertionError("data_access_class is missing")
    params = Params(trainer_config_path, **kwargs)
    data_class: BaseData = import_class(data_access_class).read(params)
    tuning_clas: BaseHyperModel = (
        HyperNetwork if build_network_from_config else import_class(tuning_class)
    )
    training_class: BaseModel = (
        import_class(training_class) if not build_network_from_config else Network
    )
    Tuner.tune(
        tuning_clas,
        training_class,
        trainer_config_path,
        hyperparameter_config_path,
        data_class,
    )
