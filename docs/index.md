# Deep Learning ML OPS

Deep Learning ML OPS, known as `dl-ml-ops`, is a kind of a ml tool to make your life easier in order to use neural network only with .yaml files. `dl-ml-ops` helps you to build network with multiple towers, multi-layer-perceptrons without touching any code, it allows us to train with given parameters. 
It also allows us to run hyperparameter tuning and serve and monitor the model performance metrics. 

Tool is uses open-source deep learning tools tensorflow, keras, keras_tunes in background. 

## Installation

Tool can be used any other package by install it via git command

```bash
poetry add git+https://github.com/caglanakpinar/dl_ml_ops.git
```

* `poetry run main.py model train --training_class ` - run model
* `poetry run main.py model tune --tuning_class ` - run model
* `poetry run main.py model serve --serving_class ` - run model [DEV]


## Project layout

    mlp/
        cli/   
            - cli.py  
        configs/
            - configurations.py  
        data_access/  
            - base.py
        logger/
            - logs.py
        monitoring/  # not yet implemented
        serve/       # not yet implemented
        train/
            - base.py
            - builder.py
            - models.py
            - trainer.py
            - tuner.py
        utils/
            - paths.py
