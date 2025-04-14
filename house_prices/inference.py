import numpy as np
import pandas as pd
from house_prices.preprocess import load_transformers
from house_prices.train import load_model


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


def make_predictions(input_data: pd.DataFrame, models_dir: str) -> np.ndarray:
    # Must match exactly what was used during training
    continuous_features = ["LotArea", "GrLivArea", "OverallQual", "YearBuilt", "BedroomAbvGr", "FullBath"]
    categorical_features = ["Neighborhood", "MSZoning"]
    
    # Ensure input data has all required columns
    required_columns = continuous_features + categorical_features
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in input data: {missing_columns}")
    
    transformers = load_transformers(models_dir)
    model = load_model(models_dir)
    
    X_test_processed = transform_data(
        input_data, continuous_features, categorical_features, transformers
    )
    return model.predict(X_test_processed)