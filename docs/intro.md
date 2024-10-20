# welcome to dl-ml-ops 101
Welcome to DL-ML-OPs wrold. Before we start, I just wanna make it clear that this platform is not a tool that you don't have any idea about the parameters about Deep Learning and make it within 10 minutes.
However this will help you to implement Neural Network smoothly. 
However, there are still need to be carefull and to learn while implementing it.
From data acress to 

## what is this all about?
This is about to build a pipeline which is starting from fetching data, build network with parameters, train tune parameters and serve

## Why do need `dl-ml-ops`?
recent years, I have been working with networks. everytime I start building it I realized that there should be some wasy to make it more generic that it can make it my life easier and I can operate it from a config file.

## Step by Step Instruction to Use
there are 8 sections;
 - configurations
 - data access
 - build network
 - train
 - store
 - hyper parameter tuning
 - serve
 - monitoring
Each section will be explained with example that are stored in `example.py`

# How to run

all comments are executed by `click.cli`. you need to call `mlp.cli.cli` to run `model.train` or `model tune`.

## Train

you can train your Deep Learning model with a .yaml file. you can write your model class by using `mlp.BaseModel` as subclass.
In order to run training;
```
poetry run python ....py model train --training_class ... --trainer_config_path ...
```

## Hyperparameter Tuning

you can tune your model by taking list of hyperparameters from a .yaml and overwrite to a train. yaml.
In order to run tuning;

```
poetry run python ....py model tune --tuning_class ... --hyherparameter_config_path ...
```

## Data Access

To be able to access a data `mlp.BaseData` will be used for class. model class will be pass to cli to run data access.
In this data class data will be generated and pass into the trainer or tuner.

argument for data access is

```
--data_class ...
```
If dataset need validation - train split, only add `split_ratio` field to .yaml at `--trainer_config_path`.
Number of inputs (features), which is dimensions of input, has to be added as field as `input_size` at `trainer_config_path`.
Number of output dimensions has to be added to `trainer_config_path` as a field as `output_size` (integer).

## Configurations

Tool only supports .yaml file as config files. there will to 2 kind of YAML file 1st for training and its folder path can be pass as arguments from `--trainer_config_path`.
2nd is for hyperparameter tuning. its folder path can be pass with argument `--tuner_config_path`.

there are mandatory fields for training yaml; 
- name
- input_size
- output_size
- target
- split_ratio

While tuning process is triggered, first, 
it takes list of parameters from `trainer_config_path`. 
Then, if any fields are matching at `hyoerparameters_config_path`, it takes those as tuning parameters. Those parameters at `hyperparameter_config_path` must be list in order to run trials for tuning.

## monitoring [DEV]

not yet implemented

## Serving [DEV]
not yet implemented