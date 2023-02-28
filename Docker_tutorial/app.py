""" Add description
"""
#%% Import libs
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
from pathlib import Path
import os

#%% API constants
ROOT_FOLDER = Path(__file__).parents[0]
MODEL_FILE = 'model/heart_disease_v1.joblib'
API_TITLE = 'Heart Disease Prediction'

#%% API modules
class InputData(BaseModel):
    """ Class with input data format and content expected by the API.
    It defines the X features required by the model. 
    The class defines the X feature name, data type and default value.
    """
    age: int = 64
    sex: int = 1 
    cp: int = 3
    trestbps: int = 120
    chol: int = 267
    fbs: int = 0
    restecg: int = 0
    thalach: int = 99
    exang: int = 1
    oldpeak: float = 1.8
    slope: int = 1
    ca: int = 2
    thal: int = 2

class OutputData(BaseModel):
    """ Class with output returned by the API. It defines the y_pred by the model.
    The class defines the y name, data type and default value.
    """
    score: float = 0.80318881046519

#%% Create API instance
app = FastAPI(title = API_TITLE)
model = load(ROOT_FOLDER / MODEL_FILE)

#%% API requests
@app.post('/score', response_model = OutputData) # POST request
def score(data: InputData):
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1,-1)
    result = model.predict_proba(model_input)[:,-1]
    
    return {'score':result}
