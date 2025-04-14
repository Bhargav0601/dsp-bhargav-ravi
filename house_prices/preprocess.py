import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess_data(
    X_train: pd.DataFrame,
    continuous_features: list,
    categorical_features: list
) -> np.ndarray:
    # Ensure all features exist in the DataFrame
    missing_continuous = [f for f in continuous_features if f not in X_train.columns]
    missing_categorical = [f for f in categorical_features if f not in X_train.columns]
    
    if missing_continuous or missing_categorical:
        raise ValueError(f"Missing features: {missing_continuous + missing_categorical}")
    
    scaler = StandardScaler()
    onehot = OneHotEncoder(handle_unknown='ignore')
    
    # Fit and transform continuous features
    X_train_continuous_scaled = scaler.fit_transform(X_train[continuous_features])
    
    # Fit and transform categorical features
    X_train_categorical_encoded = onehot.fit_transform(X_train[categorical_features])
    
    return np.concatenate(
        [X_train_continuous_scaled,
         X_train_categorical_encoded.toarray()], axis=1
    ), scaler, onehot


def save_transformers(transformers: tuple, models_dir: str) -> None:
    scaler, onehot = transformers
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(onehot, os.path.join(models_dir, 'onehot.joblib'))


def load_transformers(models_dir: str) -> tuple:
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    onehot = joblib.load(os.path.join(models_dir, 'onehot.joblib'))
    return scaler, onehot


def transform_data(
    X_test: pd.DataFrame,
    continuous_features: list,
    categorical_features: list,
    transformers: tuple
) -> np.ndarray:
    scaler, onehot = transformers
    X_test_continuous_scaled = scaler.transform(X_test[continuous_features])
    X_test_categorical_encoded = onehot.transform(X_test[categorical_features])
    return np.concatenate(
        [X_test_continuous_scaled,
         X_test_categorical_encoded.toarray()], axis=1
    )


def compute_rmsle(
    y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2
) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def load_model(models_dir: str) -> RandomForestRegressor:
    return joblib.load(os.path.join(models_dir, 'random_forest_model.joblib'))