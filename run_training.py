from house_prices.data_loader import load_data
from house_prices.train import build_model
import os

# Load data
data = load_data('train')

# Define models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

# Train and save model + transformers
results = build_model(data, models_dir)
print(f"Model trained with RMSLE: {results['rmse']}")