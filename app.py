from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("artifacts/heart_model.pkl")

@app.post("/predict")
def predict(data: dict):
    features = np.array([list(data.values())])
    prob = model.predict_proba(features)[0][1]
    pred = int(prob > 0.5)
    return {"prediction": pred, "confidence": prob}
