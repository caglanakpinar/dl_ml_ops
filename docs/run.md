# How to Run


1st, you need to create a `.py` and here is the template of it;

```
from mlp.cli.cli import cli

....


if __name__ == "__main__":
    cli()


```


2nd, you will create training configuration yaml. path of training .yaml 
will be passed from the terminal argument `--trainer_config_path`.
For more details, take a look at [hot to create configuration .yaml file](./params.md)

3rd, you need data access. To access and pass your data to tool, you need to create data access class. 
 [how to create data access](./data_access.md) will guide about  creating datasets for training and tuning.
class name and if there is a module name will passed from terminal argument `--data_access_class`.

4th, model will be ready to train after creating model training class with using `mlp.BaseModel`. 
by passing created class to the `--training_class`, framework will start training process.
for easy way, you can directly use training configuration  without using `--training_class`. 
To use default network builder from `mlp` tool, `--build_network_from_config True` has to be True.
for more details pls take a look [how to build network](./build.md) 
and for more details about training network, take a look at [how to train your network](./train.md)

Optional, you can tune your model. After tuning process is done, parameters will be updated on trainer configuration .yaml file.
First, you will create tuner configuration yaml. Path of tuner .yaml will be passed from the terminal argument `--tuner_config_path`.
For more details, take a look at [hot to create configuration .yaml file](./params.md). 
After that, model will be ready to tune after creating model tuner class with using `mlp.BaseHyperModel`.
By passing created class to the `--tuning_class`, framework will start tuning process.
for easy way, you can directly use tuning configuration  without using `--tuning_class`. 
To use default hyperparameter tuner from `mlp` tool, `--build_network_from_config True` has to be True.
for more details pls take a look [how to build network](./build.md) 
and for more details about tuner, take a look at [how to tune your network](./tune.md)

