import pickle
import pandas as pd

def model_loading():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)
    
model = model_loading()

def predict(input: pd.DataFrame):
    return model.predict(input)

def predict_proba(input: pd.DataFrame):
    return model.predict_proba(input)