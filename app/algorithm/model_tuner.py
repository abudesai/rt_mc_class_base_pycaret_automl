import os
import time
import warnings

import numpy as np
from pycaret.classification import evaluate_model, tune_model
from skopt.space import Categorical, Integer, Real

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import algorithm.model_trainer as model_trainer
import algorithm.utils as utils

# get model configuration parameters
model_cfg = utils.get_model_config()


def get_hpt_space(hpt_specs):
    param_grid = []
    for hp_obj in hpt_specs:
        if hp_obj["run_HPO"] == False:
            param_grid.append(Categorical([hp_obj["default"]], name=hp_obj["name"]))
        elif hp_obj["type"] == "categorical":
            param_grid.append(
                Categorical(hp_obj["categorical_vals"], name=hp_obj["name"])
            )
        elif hp_obj["type"] == "int" and hp_obj["search_type"] == "uniform":
            param_grid.append(
                Integer(
                    hp_obj["range_low"],
                    hp_obj["range_high"],
                    prior="uniform",
                    name=hp_obj["name"],
                )
            )
        elif hp_obj["type"] == "int" and hp_obj["search_type"] == "log-uniform":
            param_grid.append(
                Integer(
                    hp_obj["range_low"],
                    hp_obj["range_high"],
                    prior="log-uniform",
                    name=hp_obj["name"],
                )
            )
        elif hp_obj["type"] == "real" and hp_obj["search_type"] == "uniform":
            param_grid.append(
                Real(
                    hp_obj["range_low"],
                    hp_obj["range_high"],
                    prior="uniform",
                    name=hp_obj["name"],
                )
            )
        elif hp_obj["type"] == "real" and hp_obj["search_type"] == "log-uniform":
            param_grid.append(
                Real(
                    hp_obj["range_low"],
                    hp_obj["range_high"],
                    prior="log-uniform",
                    name=hp_obj["name"],
                )
            )
        else:
            raise Exception(
                f"Error creating Hyper-Param Grid. \
                Undefined value type: {hp_obj['type']} or search_type: {hp_obj['search_type']}. \
                Verify hpt_params.json file."
            )
    return param_grid


def get_default_hps(hpt_specs):
    default_hps = [hp["default"] for hp in hpt_specs]
    return default_hps


def load_best_hyperspace(results_path):
    results = [f for f in list(sorted(os.listdir(results_path))) if "json" in f]
    if len(results) == 0:
        return None
    best_result_name = results[-1]  # get maximum value
    best_result_file_path = os.path.join(results_path, best_result_name)
    return utils.get_json_file(best_result_file_path, "best_hpt_results")


def save_best_parameters(results_path, hyper_param_path):
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace(results_path)
    if space_best_model is None:
        print("No models yet. Continuing...")
        return
    print("Best model yet:", space_best_model["model_name"])
    # print("Best hyperspace yet:\n",  space_best_model["space"] )

    # Important: you must save the best parameters to /opt/ml/model/model_config/hyperparameters.json during HPO for them to persist
    utils.save_json(
        os.path.join(hyper_param_path, "hyperparameters.json"),
        space_best_model["space"],
    )


def have_hyperparams_to_tune(hpt_specs):
    for hp_obj in hpt_specs:
        if hp_obj["run_HPO"] == True:
            return True
    return False


def clear_hp_results_dir(results_path):
    if os.path.exists(results_path):
        for f in os.listdir(results_path):
            os.remove(os.path.join(results_path, f))
    else:
        os.makedirs(results_path)


def tune_hyperparameters(
    data, data_schema, num_trials, hyper_param_path, hpt_results_path
):
    # read hpt_specs file
    # hpt_specs = utils.get_hpt_specs()
    # check if any hyper-parameters are specified to be tuned
    # if not have_hyperparams_to_tune(hpt_specs):
    #     print("No hyper-parameters to tune.")
    #     return

    print("Running HPT ...")

    start = time.time()

    # clear previous results, if any
    clear_hp_results_dir(hpt_results_path)

    # set random seeds
    utils.set_seeds()
    hyper_params = utils.get_hyperparameters(hyper_param_path)
    _, base_model = model_trainer.get_trained_model(data, data_schema, hyper_params)
    opt_model = tune_model(base_model, choose_better=True, n_iter=num_trials)
    print("Optimized model:", opt_model)
    utils.save_json(
        os.path.join(hyper_param_path, "hyperparameters.json"), opt_model.get_params()
    )
    end = time.time()
    print(f"Total HPO time: {np.round((end - start)/60.0, 2)} minutes")
