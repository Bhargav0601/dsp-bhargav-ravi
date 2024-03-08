import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from house_prices.preprocess import preprocess_data, compute_rmsle


def build_model(data: pd.DataFrame) -> dict:
    (
        X_train_continuous_scaled,
        X_train_categorical_encoded,
        y_train,
        scaler,
        onehot
    ) = preprocess_data(data)
    X_train_processed = np.concatenate(
        [X_train_continuous_scaled, X_train_categorical_encoded.toarray()],
        axis=1
    )
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    joblib.dump(
        scaler,
        'C:/Users/Bhargav/dsp-bhargav-ravi/models/scaler.joblib'
    )
    joblib.dump(
        onehot,
        'C:/Users/Bhargav/dsp-bhargav-ravi/models/Encoder.joblib'
    )
    joblib.dump(
        model,
        'C:/Users/Bhargav/dsp-bhargav-ravi/models/model.joblib'
    )
    y_pred = model.predict(X_train_processed)
    rmsle = compute_rmsle(y_train, y_pred)
    return {'rmse': rmsle}
