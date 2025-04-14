from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from house_prices import load_transformers, make_predictions
import os

app = Flask(__name__)

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load model and transformers at startup
transformers = load_transformers(MODELS_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'GrLivArea': float(request.form['GrLivArea']),
            'LotArea': float(request.form['LotArea']),
            'OverallQual': int(request.form['OverallQual']),
            'YearBuilt': int(request.form['YearBuilt']),
            'Neighborhood': request.form['Neighborhood'],
            'MSZoning': request.form['MSZoning'],
            'BedroomAbvGr': int(request.form['BedroomAbvGr']),
            'FullBath': int(request.form['FullBath'])
        }
        
        # Create DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = make_predictions(input_data, MODELS_DIR)
        
        return jsonify({
            'prediction': round(prediction[0], 2),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True)