from fastapi import FastAPI
import numpy as np
import joblib
import pickle

import uvicorn

app = FastAPI(title="Customer Churn Prediction API")

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running 🚀"}

@app.post("/predict")
def predict(features: list):
    """
    Accepts a list of numeric features in the same order as training features.
    Example body:
    {
      "features": [0.3, 0.1, 1, 0, 45, 1, ...]
    }
    """
    arr = np.array(features).reshape(1, -1)
    prediction = model.predict(arr)[0]
    probability = model.predict_proba(arr)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(round(probability, 3))
    }

# Run only if directly executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
