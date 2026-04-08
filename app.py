from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np

app = Flask(__name__)

model = keras.models.load_model("final.keras")

@app.route("/")
def home():
    return "Model is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = np.array(data["input"])

    prediction = model.predict(input_data)

    return jsonify({"prediction": prediction.tolist()})

app.run()