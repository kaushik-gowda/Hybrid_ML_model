from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_lstm_model(input_shape):
    """Create and compile an LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def save_lstm_model(model, path='artifacts/models/lstm_model.h5'):
    model.save(path)
