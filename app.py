import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Production ML API")
MODEL_PATH = "artifacts/model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model artifacts missing!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
target_names = ['setosa', 'versicolor', 'virginica']

class FlowerInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: FlowerInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    return {"prediction_class_index": int(pred[0]), "predicted_flower": target_names[int(pred[0])]}
