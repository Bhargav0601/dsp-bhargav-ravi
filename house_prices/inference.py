import joblib
import numpy as np
import pandas as pd
from .preprocess import preprocess_test_data


def make_predictions(data: pd.DataFrame) -> np.ndarray:
    scaler = joblib.load(
        'C:/Users/Bhargav/dsp-bhargav-ravi/models/scaler.joblib'
    )
    onehot = joblib.load(
        'C:/Users/Bhargav/dsp-bhargav-ravi/models/Encoder.joblib'
    )
    X_test_processed = preprocess_test_data(data, scaler, onehot)
    loaded_model = joblib.load(
        'C:/Users/Bhargav/dsp-bhargav-ravi/models/model.joblib'
    )
    predicted_prices = loaded_model.predict(X_test_processed)
    return predicted_prices
