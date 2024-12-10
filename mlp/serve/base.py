import json
import threading
from abc import abstractmethod
from dataclasses import dataclass

import keras
import numpy as np
from flask import Flask, Response, request

from mlp.configs import Params
from mlp.logger import log
from mlp.train import BaseModel


@dataclass
class BaseInput:
    params: Params
    input_dict: dict = None

    @staticmethod
    def get_inputs(params: Params) -> dict:
        return params.get("serve").get("inputs")

    @classmethod
    def initialize(cls, params: Params):
        inputs = BaseInput(params)
        input_dict = BaseInput.get_inputs(params)
        for _input, _value in input_dict.items():
            setattr(inputs, _input, _value)
        setattr(inputs, "input_size", len(input_dict.items()))
        setattr(inputs, "input_dict", input_dict)
        return inputs

    def get_input_data(self, inputs: dict):
        return np.array(
            [
                inputs.get(_input, getattr(self, _input))
                for _input in self.params.get("serve").get("inputs")
            ]
        ).reshape(1, getattr(self, "input_size"))


@dataclass
class BaseOutput:
    params: Params

    @classmethod
    def initialize(cls, params: Params):
        outputs = BaseOutput(params)
        for _input, _value in params.get("serve").get("output").items():
            setattr(outputs, _input, _value)
        return outputs

    def get_output_date(self, data):
        return {
            _input: (_value if data is None else data)
            for _input, _value in self.params.get("serve").get("output").items()
        }


class CreateApi:
    def __init__(self, host=None, port=None, function=None, parameters=None):
        self.function = function
        self.parameters = parameters
        self.host = "127.0.0.1" if host is None else host
        self.port = port

    def init_api(self):
        app = Flask(__name__)
        function = self.function
        params = {p: None for p in self.parameters}

        @app.route("/")
        def render_script():
            for p in params:
                if p in request.args.keys():
                    params[p] = request.args[p]
                else:
                    params[p] = None

            heavy_process = threading.Thread(
                target=function, daemon=True, kwargs=params
            )
            heavy_process.start()
            return Response(mimetype="application/json", status=200)

        @app.route("/shutdown", methods=["POST"])
        def shutdown():
            shutdown_server()
            return "Server shutting down..."

        def shutdown_server():
            func = request.environ.get("werkzeug.server.shutdown")
            if func is None:
                raise RuntimeError("Not running with the Werkzeug Server")
            func()

        return app.run(threaded=False, debug=False, port=self.port, host=self.host)


class BaseServe:
    def __init__(self, params: Params):
        self.params = params
        self.model: keras.Model = None
        self.input_class = BaseInput.initialize(params)
        self.output_class = BaseOutput.initialize(params)
        if params.get("port") is None or params.get("host") is None:
            raise log(
                log.error,
                "<port> and <serve> fields are missing at serve_config .yaml file.",
            )
        self.port = params.get("port")
        self.host = params.get("host")

    def load_model(self):
        model = BaseModel.load(self.params)
        return model

    def init_api(self):
        app = Flask(__name__)
        params = self.input_class.get_inputs(self.params)
        function = self.serve

        @app.route("/", methods=["GET", "POST"])
        def render_script():
            data = json.loads(request.data)
            for p in params:
                if p in data.keys():
                    params[p] = data[p]

            heavy_process = threading.Thread(
                target=function, daemon=True, kwargs={"inputs": params}
            )
            heavy_process.start()
            return Response(mimetype="application/json", status=200)

        return app.run(threaded=False, debug=False, port=self.port, host=self.host)

    def predict(self, inputs):
        return self.output_class.get_output_date(
            self.model.predict(self.input_class.get_input_data(inputs), verbose=0)
        )

    @abstractmethod
    def serve(self, inputs: dict):
        NotImplementedError()
