import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_data(data):
    """Scale time series data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def prepare_lstm_data(scaled_data, time_steps=10):
    """Prepare sequences for LSTM input."""
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X).reshape(-1, time_steps, 1), np.array(y)
