import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load("artifacts/heart_model.pkl")
    sample = np.random.rand(1,13)
    pred = model.predict(sample)
    assert pred.shape == (1,)
