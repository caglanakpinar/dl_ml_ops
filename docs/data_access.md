# Data Access

To be able to access a data `mlp.BaseData` will be used for class. model class will be pass to cli to run data access.
In this data class data will be generated and pass into the trainer or tuner.

argument for data access is

```
--data_class ...
```
If dataset need validation - train split, only add `split_ratio` field to .yaml at `--trainer_config_path`.
Number of inputs (features), which is dimensions of input, has to be added as field as `input_size` at `trainer_config_path`.
Number of output dimensions has to be added to `trainer_config_path` as a field as `output_size` (integer). 
Data Access will only being used in training and tuning processes.


## How to create Data Access class by using `mlp.BaseData`

once you added `--data_class` argument in terminal framework will detect classes and will be executed.
Make sure `mlp.BaseData` will be used as sub-class. you can do whatever you want in `data_class`. 
At this data access class, there 2 abstract methods has to be added to `read`and `__init__`. Both will take params as argument.
Framework needs `read` function. `read` function always takes `params`. It will be executed as `classmethod` and will return your data access class
Within `read` function, your data access clas will be created. Data will be fetched regarding your data source (wherever from).
this fetched dataset has to be assigned `data` attribute into your data access class
Here is an example of how you can build your own data access class. 

```
from mlp import BaseData, Params


class MyDataAccess(BaseData)
    def __init__(self, params: Params):
        self.params = params
        ...
        ...
        
     @classmethod
     def read(cls, params: Params):
        _cls = MyDataAccess(params)
        ....
        ....
        _cls.data = <dont forget to assign raw data to <data> attribute>
        return _cls
```


## Example of creating Data Access class

Here an example of treading data from `https://storage.googleapis.com/tf-datasets/titanic/train.csv` which is open-source dataset as titanic survivors.
We would like to read this data as pandas data fram in our data access class

```
class MyBinaryClassificationData(BaseData):
    """
    we will be using titanic csv file from
    https://storage.googleapis.com/tf-datasets/titanic/train.csv
    """

    def __init__(self, params: Params):
        self.params = params

    @classmethod
    def read(cls, params: Params, **kwargs):
        titanic_file = keras.utils.get_file("train.csv", params.get("data_url"))
        _cls = MyBinaryClassificationData(params)
        _cls.data = pd.read_csv(titanic_file)
        log(log.info, "One Hot Encoding for Categorical Features ...")
        for categorical_column in [
            "sex",
            "n_siblings_spouses",
            "parch",
            "class",
            "deck",
            "embark_town",
        ]:
            _dummies = pd.get_dummies(
                _cls.data[categorical_column], dtype=int, prefix=categorical_column
            )
            _cls.data = _cls.data.drop(categorical_column, axis=1)
            _cls.data = pd.concat([_cls.data, _dummies], axis=1)
        _cls.data["alone"] = _cls.data.alone.apply(lambda x: 0 if x == "n" else 1)
        return _cls
```
