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
        ''' 
        we use the 'self.class_name_prefix' to alter the given class names (labels). 
        More specifically, we concatenate with the prefix to ensure class names are strings. 
        When classes are already strings, this is not needed. 
        When pycaret sees numerical class names, it assumes the user has already encoded them as integers starting
        at zero. This causes an exception when you have classes that dont start at zero, which is the case
        with page blocks dataset.
        '''
        self.class_name_prefix = "__c__"

    def fit(self, train_data, _categorical, _numerical):
        self.class_names = sorted(
            train_data[self.target_field].drop_duplicates().tolist()
        )

        train_data[self.target_field] = train_data[self.target_field].apply(
            lambda c: f"{self.class_name_prefix}{c}"
        )

        self._categorical = _categorical
        self._numerical = _numerical

        all_features = self._categorical + self._numerical + [self.target_field]
        setup(
            data=train_data[all_features],
            target=self.target_field,
            categorical_features=_categorical if len(_categorical) > 0 else None,
            numeric_features=_numerical if len(_numerical) > 0 else None,
            silent=True,
            verbose=False,
            session_id=42,
            data_split_stratify=True
        )

        best_model = compare_models(verbose=False)
        self.model = best_model

        metrics = pull()
        print(metrics)
        return self


    def predict_proba(self, X):
        predictions = predict_model(self.model, X, raw_score=True)
        """ pycaret returns a dataframe with 1 + c columns added to the end, where c is number of classes:
            'Label' which has the predicted class
            'Score_<class_0>' which has the predicted probability for the class with name "class_0"
            'Score_<class_1>' which has the predicted probability for the class with name "class_1"
            .
            .

            Below we will get the grab and return the probability columns
        """    
        score_columns = [f"Score_{self.class_name_prefix}{c}" for c in self.class_names]
        predictions = predictions[score_columns]
        predictions.columns = self.class_names  # rename with original class names
        return predictions[self.class_names]

    def predict(self, X):
        preds = predict_model(self.model, X[self._categorical + self._numerical])
        return preds[["Label"]]


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
