from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


def build_lstm(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def save_lstm_model(model, path='artifacts/models/lstm_model.h5'):
    model.save(path)
