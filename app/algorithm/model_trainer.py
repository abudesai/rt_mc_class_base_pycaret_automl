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

    resampled_data = get_resampled_data(train_data, target_field)
    model = classifier.fit(resampled_data, _categorical, _numerical)
    return model



def get_resampled_data(data, target_field):

    # if some minority class is observed only 1 time, and a majority class is observed 100 times
    # we dont over-sample the minority class 100 times. We have a limit of how many times
    # we sample. max_resample is that parameter - it represents max number of full population
    # resamples of the minority class. For this example, if max_resample is 3, then, we will only
    # repeat the minority class 2 times over (plus original 1 time).
    y = data[target_field]
    max_resample = model_cfg["max_resample_of_minority_classes"]
    unique, class_count = np.unique(y, return_counts=True)
    # class_count = [ int(c) for c in class_count]
    max_obs_count = max(class_count)

    resampled_data = []
    for i, count in enumerate(class_count):
        if count == 0:
            continue
        # find total num_samples to use for this class
        size = (
            max_obs_count
            if max_obs_count / count < max_resample
            else count * max_resample
        )
        # if observed class is 50 samples, and we need 125 samples for this class,
        # then we take the original samples 2 times (equalling 100 samples), and then randomly draw
        # the other 25 samples from among the 50 samples

        full_samples = size // count
        idx = y == unique[i]
        for _ in range(full_samples):
            resampled_data.append(data.loc[idx])

        # find the remaining samples to draw randomly
        remaining = size - count * full_samples
        idx_list = list(data.loc[idx].index)
        sampled_idx = np.random.choice(idx_list, size=remaining, replace=True)
        resampled_data.append(data.iloc[sampled_idx])

    resampled_data = pd.concat(resampled_data, axis=0, ignore_index=True)

    # shuffle the arrays
    resampled_data = shuffle(resampled_data)
    return resampled_data
