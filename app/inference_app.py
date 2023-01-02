# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

import io
import numpy as np
import os
import sys
import traceback
import warnings
from tempfile import NamedTemporaryFile
from typing import Union

import pandas as pd
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

warnings.filterwarnings("ignore")
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"


import algorithm.utils as utils
from algorithm.model.classifier import MODEL_NAME
from algorithm.model_server import ModelServer

prefix = "/opt/ml_vol/"
data_schema_path = os.path.join(prefix, "inputs", "data_config")
model_path = os.path.join(prefix, "model", "artifacts")
failure_path = os.path.join(prefix, "outputs", "errors", "serve_failure")


# get data schema - its needed to set the prediction field name
# and to filter df to only return the id and pred columns
data_schema = utils.get_data_schema(data_schema_path)


# initialize your model here before the app can handle requests
model_server = ModelServer(model_path=model_path, data_schema=data_schema)


# The FastAPI app for serving predictions
app = FastAPI()


async def gen_temp_file(ext: str = ".csv"):
    """Generate a temporary file with a given extension"""
    with NamedTemporaryFile(suffix=ext, delete=True) as temp_file:
        yield temp_file.name


@app.get("/ping", tags=["ping", "healthcheck"])
async def ping() -> dict:
    """Determine if the container is working and healthy."""
    response = f"Hello, I am {MODEL_NAME} model and I am at you service!"
    return {
        "success": True,
        "message": response,
    }


@app.post("/infer", tags=["inference", "json-infer"], response_class=JSONResponse)
async def infer(input_: dict) -> dict:
    """Generate inferences on a single batch of data sent as JSON object.
    In this sample server, we take data as JSON, convert
    it to a pandas data frame for internal use and then convert the predictions back to JSON .
    """
    try:
        # Do the prediction
        data = pd.DataFrame.from_records(input_["instances"])
        print(f"Invoked with {data.shape[0]} records")
        predictions_response = model_server.predict_to_json(data)
        return {
            "success": True,
            "predictions": predictions_response,
        }

    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        with open(failure_path, "w") as s:
            s.write("Exception during inference: " + str(err) + "\n" + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during inference: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        return {
            "success": False,
            "message": f"Exception during inference: {str(err)} (check failure file for more details)",
        }
