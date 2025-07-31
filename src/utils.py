import os
import joblib
import yaml
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys


def load_data(path='notebook/data/apple_stock.csv'):
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)


def prepare_data(df, look_back=10, test_size=0.2):
    logging.info("Preparing time-series data...")


def add_lag_features(data, lags=[1, 2, 3]):
    """Create lag features for time series data."""
    try:
        for lag in lags:
            data[f'Lag_{lag}'] = data['Close'].shift(lag)
        return data.dropna()
    except Exception as e:
        raise CustomException(e, sys)


def train_test_split_time_series(X, y, train_ratio=0.8):
    """Split time series data without shuffling."""
    try:
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, sys)


def save_model(model, filepath):
    """Save model to disk using joblib."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logging.info(f"Model saved at: {filepath}")
    except Exception as e:
        raise CustomException(e, sys)


def load_model(filepath):
    """Load model from disk using joblib."""
    try:
        model = joblib.load(filepath)
        logging.info(f"Model loaded from: {filepath}")
        return model
    except Exception as e:
        raise CustomException(e, sys)


def load_config(path='config.yaml'):
    """Load a YAML configuration file."""
    try:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info("Configuration file loaded successfully.")
            return config
    except Exception as e:
        raise CustomException(e, sys)
