import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.logger import logging  # Make sure logger is working


def create_lstm_model(input_shape):
    logging.info("Creating LSTM model...")
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    logging.info("LSTM model created successfully.")
    return model


def train_lstm_model(X_train, y_train, X_val, y_val,
                     model_name="lstm_model.keras"):
    logging.info("Training LSTM model...")

    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = create_lstm_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Save model to artifacts folder
    artifacts_dir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, model_name)
    model.save(model_path)
    logging.info(f"LSTM model saved to {model_path}")
    return model
