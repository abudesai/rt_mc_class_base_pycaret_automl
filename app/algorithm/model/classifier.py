import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from pycaret.classification import compare_models
from pycaret.classification import load_model as import_model
from pycaret.classification import predict_model, pull
from pycaret.classification import save_model as dump_model
from pycaret.classification import setup

warnings.filterwarnings("ignore")


model_fname = "model.save"
pipeline_fname = "pipeline.save"
MODEL_NAME = "mc_class_base_pycaret"


class Classifier:
    def __init__(self, id_field, target_field, **kwargs) -> None:
        self.target_field = target_field
        self.id_field = id_field
        self.class_names = []

    def fit(self, train_data, _categorical, _numerical):
        self.class_names = sorted(
            train_data[self.target_field].drop_duplicates().tolist()
        )
        self._categorical = _categorical
        self._numerical = _numerical

        all_features = [self.target_field] + self._categorical + self._numerical
        setup(
            data=train_data[all_features],
            target=self.target_field,
            categorical_features=_categorical,
            numeric_features=_numerical,
            silent=True,
            verbose=False,
            session_id=42,
        )

        best_model = compare_models(verbose=False)

        self.model = best_model
        metrics = pull()
        print(metrics)
        return self

    def predict(self, X, verbose=False):
        preds = predict_model(self.model, X[self._categorical + self._numerical])
        return preds[["Label"]]

    def predict_proba(self, X, verbose=False):

        predictions = predict_model(self.model, X[self._categorical + self._numerical])

        """ pycaret returns a dataframe with two columns added to the end:
            'Label' which has the predicted class
            'Score' which has the predicted probability for the predicted class

            Below we will process it into a dataframe with column headers as class names, and values
            as the probabilities
        """
        # start with zero probabilities
        processed_predictions = pd.DataFrame(
            np.zeros(shape=(predictions.shape[0], len(self.class_names))), columns=self.class_names
        )
        # populate the probabilities for the two classes
        for i, c in enumerate(self.class_names):
            idx = predictions["Label"] == c
            if i == 0:
                processed_predictions.loc[idx, self.class_names[0]] = predictions.loc[
                    idx, "Score"
                ]
                processed_predictions.loc[idx, self.class_names[1]] = (
                    1 - predictions.loc[idx, "Score"]
                )
            else:
                processed_predictions.loc[idx, self.class_names[1]] = predictions.loc[
                    idx, "Score"
                ]
                processed_predictions.loc[idx, self.class_names[0]] = (
                    1 - predictions.loc[idx, "Score"]
                )
        return processed_predictions

    def summary(self):
        self.model.get_params()

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))
        dump_model(self.model, os.path.join(model_path, pipeline_fname))

    @classmethod
    def load(cls, model_path):
        classifier = joblib.load(os.path.join(model_path, model_fname))
        classifier.model = import_model(os.path.join(model_path, pipeline_fname))
        return classifier


def save_model(classifier, model_path):
    classifier.save(model_path)


def load_model(model_path):
    classifier = Classifier.load(model_path)
    return classifier
