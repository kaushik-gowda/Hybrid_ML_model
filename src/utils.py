import pandas as pd
import numpy as np
import os
import logging
import joblib
import yaml


def add_lag_features(data, lags=[1, 2, 3]):
    """Create lag features for time series data."""
    for lag in lags:
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    return data.dropna()


def train_test_split_time_series(X, y, train_ratio=0.8):
    """Split time series data without shuffling."""
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test


def setup_logger(name=__name__, log_file='logs/app.log', level=logging.INFO):
    """Set up a logger that writes to a file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def save_model(model, filepath):
    """Save model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model from disk using joblib."""
    return joblib.load(filepath)


def load_config(path='config.yaml'):
    """Load a YAML configuration file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)
