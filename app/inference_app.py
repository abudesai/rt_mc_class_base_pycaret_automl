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
        predictions_df = model_server.predict_proba(data)
        predictions_df.columns = [str(c) for c in predictions_df.columns]
        class_names = predictions_df.columns[1:]

        predictions_df["__label"] = pd.DataFrame(
            predictions_df[class_names], columns=class_names
        ).idxmax(axis=1)

        # convert to the json response specification
        id_field_name = model_server.id_field_name
        predictions_response = []
        for rec in predictions_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[id_field_name] = rec[id_field_name]
            pred_obj["label"] = rec["__label"]
            pred_obj["probabilities"] = {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [id_field_name, "__label"]
            }
            predictions_response.append(pred_obj)

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


@app.post(
    "/infer_file", tags=["inference_file", "csv-infer"], response_class=FileResponse
)
async def infer_file(
    input_: UploadFile = File(...), temp=Depends(gen_temp_file)
) -> Union[FileResponse, dict]:
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    print(input_)
    if input_.content_type == "text/csv":
        data = await input_.read()
        temp_io = io.StringIO(data.decode("utf-8"))
        data = pd.read_csv(temp_io)
        print(data.head())
    else:
        return {
            "success": False,
            "message": f"Content type {input_.content_type} not supported (only CSV data allowed)",
        }

    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    try:
        predictions = model_server.predict(data)
        # Convert from dataframe to CSV
        predictions.to_csv(temp, index=False)
        return FileResponse(temp, media_type="text/csv")
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
