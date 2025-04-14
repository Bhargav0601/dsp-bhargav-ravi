# house_prices/data_loader.py
import pandas as pd
import os

def load_data(data_type='train'):
    """
    Load the training or test dataset
    Args:
        data_type (str): 'train' or 'test'
    Returns:
        pd.DataFrame: Loaded dataset
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if data_type == 'train':
        file_path = os.path.join(data_dir, 'train.csv')
    elif data_type == 'test':
        file_path = os.path.join(data_dir, 'test.csv')
    else:
        raise ValueError("data_type must be either 'train' or 'test'")
    
    return pd.read_csv(file_path)