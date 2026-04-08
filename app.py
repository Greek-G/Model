from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Lovable frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model once at startup
model = tf.keras.models.load_model("final.keras")

# Your class labels — update this list to match your model's output
CLASS_NAMES = [
    "Healthy",
    "Leaf Blight",
    "Leaf Rust",
    # ... add all your classes here
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))  # adjust size to match your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "disease": CLASS_NAMES[predicted_index],
        "confidence": round(confidence * 100, 2)
    }

@app.get("/")
def health_check():
    return {"status": "ML API is running"}