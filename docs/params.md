# Configurations

Tool is fully configurable from argument that has been passed from terminal or configurable .yaml files.
However, there are still rules and mandatory fields to be carefully while designing a pipeline.

Tool only supports .yaml file as config files. 
there will be 2 kind of YAML files. 1st for training and it is the  folder path can be pass as arguments from `--trainer_config_path`.
2nd is for hyperparameter tuning. it is the folder path can be pass with argument `--tuner_config_path`.

If extra fields are passed from the terminal, still, it can be used within framework which will be stored in to configuration class `Params`.

## `Params` object

This takes all fields from .yaml file and adds as a new attribute into the `Params`. 
When it is needed to call an attribute from it, `params.get('<name of the field>')`
There are 2 types of .yaml can be sorted into the params classes; 

- trainer configurations; from `--trainer_config_path`
- tuner configurations; from `--tuner_config_path`

If extra fields are passed from the terminal still it can be used within frame which will be stored in to configuration class `Params`.
let's say we throw below arguments from terminal;

```
--l1_regulization_term 0.001 --l2_regulization_term 0.001
```
these arguments `l1_regulization_term` and `l2_regulization_term` are not used as default arguments while training or tuning.
However, it can be passed to `Params` class from cli from below code;

```
  
cli.py
....
    
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
    ...
    ....
    
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
```
all arguments from terminal will be captured and available on `kwargs` on cli. 


### Specific Training Arguments

Above fields are mandatory depending on your model architect. 

```
name: "give a name to model this will be used anywehere such as chekpoint save, etc"
input_size: "mandotary when --build_network_from_config True"
output_size: "mandotary when --build_network_from_config True"
target: "mandotary when --build_network_from_config True and it is supervised"
split_ratio: "mandotary"
metrics: "metrics that will be used for "
```

### Specific tuning Arguments

Above fields are mandatory depending on your model architect. 

```
max_trials: "number of maximum trials for tune parameterss"
```

## Example Training Configurations

example for training .yaml file 
```
activation: relu
activation_output: sigmoid
batch_size: 64
checkpoint_monitor: accuracy
checkpoint_save_frequency: 5
data_url: https://storage.googleapis.com/tf-datasets/titanic/train.csv
dropout: 0.0
epochs: 10
h_layers: 3
input_size: 33
l1: 0.0001
l2: 0.0001
loss: binary_crossentropy
lr: 0.0004
metrics:
- accuracy
name: BINARY_CLASSIFICATION
optimizer: adam
output_size: 1
target: alone
units: 16
use_bias: false

```

each fields above will be added to `Params` class.


## Example of Tuning Configurations

fields are in tuning .yaml file has to be found in training .yaml file.
each field of values must return list. each trial selected value from the list will be used for tuning.

```
lr:
  - 0.0001
  - 0.0002
  - 0.0003
  - 0.0004
  - 0.0005
max_trials: 5
h_layers:
  - 5
  - 4
  - 3
```

`lr` and `h_layers` will be tuned regarding of given list of values.
