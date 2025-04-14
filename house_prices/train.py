import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from house_prices.preprocess import load_model, preprocess_data, transform_data
from house_prices.preprocess import split_data, save_transformers
from house_prices.preprocess import load_transformers, compute_rmsle


def train_model(
    X_train_processed: np.ndarray,
    y_train: pd.Series,
    models_dir: str
) -> RandomForestRegressor:
    model = RandomForestRegressor()
    model.fit(X_train_processed, y_train)
    joblib.dump(model, os.path.join(models_dir, 'random_forest_model.joblib'))
    return model


def build_model(data: pd.DataFrame, models_dir: str) -> dict:
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Update these lists to include all features we want to use
    continuous_features = ["LotArea", "GrLivArea", "OverallQual", "YearBuilt", "BedroomAbvGr", "FullBath"]
    categorical_features = ["Neighborhood", "MSZoning"]
    
    X_train_processed, scaler, onehot = preprocess_data(
        X_train, continuous_features, categorical_features
    )
    save_transformers((scaler, onehot), models_dir)
    model = train_model(X_train_processed, y_train, models_dir)
    
    transformers = load_transformers(models_dir)
    model = load_model(models_dir)
    X_test_processed = transform_data(
        X_test, continuous_features, categorical_features, transformers
    )
    y_pred = model.predict(X_test_processed)
    rmsle = compute_rmsle(y_test, y_pred)
    return {'rmse': rmsle}