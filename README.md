# House Price Prediction Project

## Overview
This project is a machine learning-based solution for predicting house prices using property features. The system includes data preprocessing, model training, and a Flask-based web application for making predictions. The model is trained on housing data and can predict prices based on features like lot area, living area, neighborhood, and year built.

## Live Demo
Access the live demo here: [House Price Prediction App](https://house-price-pl2a.onrender.com)

## Features
- **Machine Learning Model**: Uses RandomForestRegressor for price predictions
- **Web Interface**: User-friendly Flask web application
- **Data Processing**: Handles both numerical and categorical features
- **Performance Metrics**: Evaluates model using Root Mean Squared Log Error (RMSLE)

## Project Structure
```
house-price-prediction/
├── data/                   # Dataset files
│   ├── train.csv           # Training data
│   └── test.csv            # Test data
├── models/                 # Saved models and transformers
│   ├── model.joblib        # Trained model
│   ├── scaler.joblib       # Feature scaler
│   └── encoder.joblib      # Categorical encoder
├── notebooks/              # Jupyter notebooks
│   ├── House_prediction.ipynb
│   ├── model-industrialization-1.ipynb
│   └── model-industrialization-final.ipynb
├── house_prices/           # Python package
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── train.py            # Model training
│   ├── inference.py        # Prediction functions
│   └── preprocess.py       # Data preprocessing
├── app.py                  # Flask application
├── templates/              # HTML templates
│   └── index.html          # Web interface
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Training the model**:
   ```python
   from house_prices.train import build_model
   import pandas as pd
   
   data = pd.read_csv('data/train.csv')
   performance = build_model(data, models_dir='models')
   print(performance)
   ```

2. **Making predictions**:
   ```python
   from house_prices.inference import make_predictions
   import pandas as pd
   
   input_data = pd.DataFrame({
       'LotArea': [8450],
       'GrLivArea': [1710],
       'OverallQual': [7],
       'YearBuilt': [2003],
       'Neighborhood': ['NAmes'],
       'MSZoning': ['RL']
   })
   
   predictions = make_predictions(input_data, models_dir='models')
   print(predictions)
   ```

3. **Running the web app**:
   ```bash
   python app.py
   ```
   Then visit `http://localhost:5000` in your browser.

## Dependencies
- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

## Model Performance
The model achieves an RMSLE (Root Mean Squared Log Error) of approximately 0.24 on the test set.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.