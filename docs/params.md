# Configurations

Tool is fully configurable from argument that has been passed from terminal or configurable .yaml files.
However, there are still rules and mandatory fields to be carefully while designing a pipeline.

Tool only supports .yaml file as config files. there will to 2 kind of YAML file 1st for training and its folder path can be pass as arguments from `--trainer_config_path`.
2nd is for hyperparameter tuning. its folder path can be pass with argument `--tuner_config_path`.

## Arguments

### Specific Training Arguments

### Specific tunşng Arguments

## Training Configurations

```
name: "give a name to model this will be used anywehere such as chekpoint save, etc"
input_size: "mandotary when --build_network_from_config True"
output_size: "mandotary when --build_network_from_config True"
target: "mandotary when --build_network_from_config True and it is supervised"
split_ratio: "mandotary"

```

## tuning Configurations


