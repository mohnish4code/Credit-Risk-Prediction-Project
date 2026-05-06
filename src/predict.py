import pickle
import os
import pandas as pd

def model_loading():
    path = "models/model.pkl"
    print("Loading model from:", os.path.abspath(path))
    
    with open(path, "rb") as f:
        return pickle.load(f)

    
model = model_loading()

def predict(input: pd.DataFrame):
    return model.predict(input)

def predict_proba(input: pd.DataFrame):
    return model.predict_proba(input)