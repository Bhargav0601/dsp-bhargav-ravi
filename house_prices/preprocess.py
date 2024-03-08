import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_log_error
import numpy as np


def preprocess_data(data: pd.DataFrame) -> tuple:
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    continuous_features = ["LotArea", "YearBuilt"]
    categorical_features = ["Neighborhood", "BldgType"]
    scaler = StandardScaler()
    onehot = OneHotEncoder(handle_unknown='ignore')
    X_train_continuous_scaled = scaler.fit_transform(
        X_train[continuous_features]
    )
    X_train_categorical_encoded = onehot.fit_transform(
        X_train[categorical_features]
    )
    return (X_train_continuous_scaled,
            X_train_categorical_encoded,
            y_train,
            scaler,
            onehot)


def preprocess_test_data(
    data: pd.DataFrame, scaler: StandardScaler, onehot: OneHotEncoder
) -> np.ndarray:
    continuous_features = ["LotArea", "YearBuilt"]
    categorical_features = ["Neighborhood", "BldgType"]
    X_test_continuous_scaled = scaler.transform(
        data[continuous_features]
    )
    X_test_categorical_encoded = onehot.transform(
        data[categorical_features]
    )
    X_test_processed = np.concatenate(
        [
            X_test_continuous_scaled, X_test_categorical_encoded.toarray()
        ], axis=1
    )
    return X_test_processed


def compute_rmsle(
    y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2
) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)
