#!/usr/bin/env python

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pprint

import algorithm.utils as utils
import numpy as np
import pandas as pd
from algorithm.model.classifier import Classifier
from algorithm.utils import get_model_config
from sklearn.utils import shuffle

# get model configuration parameters
model_cfg = get_model_config()


def get_trained_model(train_data, data_schema, hyper_params):

    # set random seeds
    utils.set_seeds()

    # print('train_data shape:',  train_data.shape)

    # Create and train model
    print("Fitting model ...")
    model = train_model(
        train_data,
        data_schema,
        hyper_params,
    )

    return model


def train_model(train_data, data_schema, hyper_params):
    info = data_schema["inputDatasets"]["multiClassClassificationBaseMainInput"]
    target_field = info["targetField"]
    id_field = info["idField"]
    _categorical = [
        c["fieldName"]
        for c in info["predictorFields"]
        if c["dataType"] == "CATEGORICAL"
    ]
    _numerical = [
        c["fieldName"] for c in info["predictorFields"] if c["dataType"] == "NUMERIC"
    ]

    classifier = Classifier(id_field, target_field, **hyper_params)
    model = classifier.fit(train_data, _categorical, _numerical)
    return model
