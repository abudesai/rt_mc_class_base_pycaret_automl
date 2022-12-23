import os
import sys

import algorithm.model.classifier as classifier
import algorithm.utils as utils
import numpy as np

# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"][
            "multiClassClassificationBaseMainInput"
        ]["idField"]
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = classifier.load_model(self.model_path)
        return self.model

    def predict(self, data):
        model = self._get_model()

        if model is None:
            raise Exception("No model found. Did you train first?")

        # make predictions
        preds = model.predict(data)

        preds_df = data[[self.id_field_name]].copy()
        preds_df["prediction"] = preds.values

        return preds_df

    def predict_proba(self, data):
        print(data.shape)
        preds = self._get_predictions(data)
        preds_df = data[[self.id_field_name]].copy()
        for c in preds.columns:
            preds_df[c] = preds[c]

        return preds_df

    def _get_predictions(self, data):
        model = self._get_model()
        preds = model.predict_proba(data)
        return preds
