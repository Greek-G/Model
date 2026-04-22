from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model("final.keras")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Model running on Render"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": pred.tolist()}