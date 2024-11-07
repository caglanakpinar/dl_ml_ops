# Train

Training process will be executed by the Tool after given arguments from the terminal.
In order to execute training process

## Trainer Class

You can directly use your own Network to train
when you will use your own trainer class, you have to pass;
```
--training_class examples.MyBinaryClassificationModel
```

In above example, framework will import `MyBinaryClassificationModel` from `examples`
and will fit the model which is already in `MyBinaryClassificationModel`. Model has to be assigned to `self.model` in the class.

In the training class there are some rules;

- under training class, `mlp.BaseModel` will be used as sub-class.
- there are some abstractmethod will be used within trainer class that are coming from `mlp.BaseModel`
  - `__init__(self, params: mlp.Params)`: initialize your training class and don't forget to add aguments as `params` there.
`params` will be coming from trainer config path which you have to pass from terminal in `--tariner_config_path`

```
from mlp import BaseModel, Params


class MyBinaryClassificationModel(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        ....
```
- Another abstractmethod that is coming from `mlp.BaseModel`, is `train`. In order to train your model trainer attribute will be needed.
This method will need arguments `dataset: BaseData.data_type`. you can use `dataset` in your model. 
How to use data set will be in [dataset section](./data_access.md).

```
from mlp import BaseModel, Params


class MyBinaryClassificationModel(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        ....
        
    def train(self, dataset: BaseData.data_type) -> None:
        ....
        ....
```
- Model has to be built within trainer class. It has to be assigned to `self.model`. Where ever you need Network
`self.model` attribute will be called. `self.model` has to be a `keras.Model`.
In below example, model will be created at `__init__` from `buildv2`;

```
from mlp import BaseModel, Params


class MyBinaryClassificationModel(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        self.model = buildv2()
        
    def buildv2(self) -> keras.Model:
        ....
        return keras.Model()
```

Let's take a look at the below example;

```
class MyBinaryClassificationModel(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        self.model = self.buildv1()

    def metrics(self):
        return [Metrics.train_epoch_metrics(metric) for metric in self.params.metrics]

    def buildv1(self) -> keras.Model:
        h_units = BaseModel.cal_hidden_layer_of_units(
            self.params.h_layers, self.params.units
        )
        _input = layers.Input(
            name=f"{self.params.model_type}_{self.params.name}_input",
            shape=(self.params.input_size,),
        )
        _hidden = layers.BatchNormalization()(_input)
        for _unit in h_units:
            _hidden = layers.Dense(
                _unit,
                activation=BaseModel.decision_of_activation(self.params.activation),
                use_bias=False,
            )(_hidden)
            _hidden = layers.Dropout(self.params.dropout)(_hidden)
        output = layers.Dense(
            self.params.output_size,
            name="output",
            activation=BaseModel.decision_of_activation(self.params.activation_output),
            use_bias=self.params.use_bias,
            kernel_regularizer=regularizers.l1_l2(l1=self.params.l1, l2=self.params.l2),
        )(_hidden)
        model = keras.Model(inputs=_input, outputs=output)
        model.compile(
            loss=self.params.loss,
            optimizer=self.optimizer(self.params.optimizer, self.params.lr),
            metrics=self.metrics(),
        )
        return model
```

Model will be trained by `MyBinaryClassificationModel`. As you see that Network has been created by `buildv1`.
You can use this template to build your own network.

Another approach is that you can use your build-in classes to create your own network. In below example,
network will be built by using `mlp.Network` class. You don't needd to use a trainer class to call `mlp.Network`. 
By only passing `trainer_config_path` model will create network by using `mlp.Network`.


```
class MyBinaryClassificationModelV2(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        self.model = self.build()

    def build(self):
        from mlp import Network
        network = Network(params=self.params)
        network.build_network_from_config()
        return network.model

    def train(self, dataset: BaseData.data_type):
        self.model.fit(
            # (True (train-val split), True (y variable available for this data), x (we are taking INPUT variables))
            x=dataset[(True, True, "x")],
            # (True (train-val split), True (y variable available for this data), x (we are taking TARGET variable))
            y=dataset[(True, True, "y")],
            # (True (train-val split), True (y variable available for this data),
            # validation_data (we are taking validation data tuple(x, y) ))
            validation_data=dataset[(True, True, "validation_data")],
            batch_size=self.params.batch_size,
            epochs=self.params.epochs,
        )

```


## Train from Configuration

You can directly use built-in modules to create Network


