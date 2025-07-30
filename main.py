import pandas as pd
import numpy as np
import os
import tensorflow as tf
from src.data.load_data import load_stock_data
from src.preprocessing.scale_and_sequence import prepare_lstm_data, scale_data
from src.models.lstm_model import create_lstm_model
from src.models.linear_model import train_linear_model
from src.prediction.forecast_lstm import forecast_with_lstm
from src.prediction.forecast_linear import forecast_with_linear
from src.prediction.forecast_hybrid import forecast_with_hybrid
from src.visualization.plot_results import plot_predictions
from src.utils import add_lag_features, train_test_split_time_series
from src import logger
from src.exception import CustomException
import logging
import sys


# Get path to the current log file
log_file_path = logger.get_log_path()

# Open the log file in append mode
log_file = open(log_file_path, "a")

# Redirect stdout and stderr to log file
os.dup2(log_file.fileno(), sys.stdout.fileno())
os.dup2(log_file.fileno(), sys.stderr.fileno())

# Optional: Limit TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":
    logging.info("Main script started.")


def main():
    try:
        # Load data
        df = load_stock_data('data/apple_stock.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Linear Model Preparation
        df_lag = add_lag_features(df.copy(), lags=[1, 2, 3])
        X = df_lag[['Lag_1', 'Lag_2', 'Lag_3']]
        y = df_lag['Close']
        X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

        # Train Linear Model
        lin_model = train_linear_model(X_train, y_train)
        lin_predictions = lin_model.predict(X_test)

        # Scale and prepare data for LSTM
        scaled_data, scaler = scale_data(df[['Close']])
        X_lstm, y_lstm = prepare_lstm_data(scaled_data)
        split_index = int(len(X_lstm) * 0.8)
        X_lstm_train, y_lstm_train = X_lstm[:split_index], y_lstm[:split_index]
        X_lstm_test, y_lstm_test = X_lstm[split_index:], y_lstm[split_index:]

        # Train LSTM Model
        lstm_model = create_lstm_model((X_lstm_train.shape[1], 1))
        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10,
                       batch_size=16, verbose=0)

        # Forecasting
        lstm_predictions = forecast_with_lstm(lstm_model, X_lstm_test, scaler)
        lin_future_predictions = forecast_with_linear(lin_model,
                                                      df['Close'].values[-3:],
                                                      scaler)
        hybrid_predictions = forecast_with_hybrid(lstm_model, lin_model,
                                                  df['Close'].values[-3:],
                                                  scaler)

        # Dates for future predictions
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=10)

        # Build forecast DataFrame
        future_df = pd.DataFrame({
            'Date': future_dates,
            'LSTM Predictions': lstm_predictions.flatten(),
            'Linear Regression Predictions': lin_future_predictions.flatten(),
            'Hybrid Model Predictions': hybrid_predictions.flatten()
        })

        print("\nForecast Results:")
        print(future_df)

        # Plotting
        plot_predictions(df, lstm_predictions, lin_predictions, future_df)

    except Exception as e:
        raise CustomException(e, sys)
