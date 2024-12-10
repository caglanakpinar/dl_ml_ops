from mlp.serve.base import BaseServe, Params


class ServeNetwork(BaseServe):
    def __init__(self, params: Params):
        super().__init__(params)
        self.model = self.load_model()

    def serve(self, inputs: dict):
        predictions = self.predict(inputs)
        return self.output_class.get_output_date(predictions)
