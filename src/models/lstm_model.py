import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.logger import logging


def create_lstm_model(input_shape):
    logging.info("Creating LSTM model...")
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_model(X_train, y_train, X_val, y_val,
                     model_name="lstm_model.keras"):
    logging.info("Training LSTM model...")

    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    artifacts_dir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, model_name)
    model.save(model_path)
    logging.info(f"LSTM model saved to {model_path}")
    return model
