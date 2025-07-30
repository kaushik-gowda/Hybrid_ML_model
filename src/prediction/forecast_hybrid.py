import numpy as np
import pandas as pd


def forecast_with_hybrid(lstm_model, linear_model, recent_data_lstm,
                         recent_data_linear, n_future, scaler):
    """
    Forecast future values using both LSTM and Linear models, then average 
    them.
    
    Args:
        lstm_model: Trained LSTM model.
        linear_model: Trained linear regression model.
        recent_data_lstm: Recent LSTM input data (3D for LSTM).
        recent_data_linear: Recent linear input data (1D or 2D).
        n_future: Number of time steps to forecast.
        scaler: Scaler used to inverse transform predictions.

    Returns:
        Array of forecasted values (inverse transformed).
    """
    lstm_predictions = []
    linear_predictions = []
    hybrid_predictions = []

    current_input_lstm = recent_data_lstm.copy()
    current_input_linear = recent_data_linear.copy()

    for _ in range(n_future):
        # LSTM prediction
        lstm_pred = lstm_model.predict(current_input_lstm.reshape(
            1,
            current_input_lstm.shape[0],
            current_input_lstm.shape[1]), verbose=0)[0, 0]
        lstm_predictions.append(lstm_pred)

        # Linear prediction
        linear_pred = linear_model.predict(
            current_input_linear.reshape(1, -1))[0]
        linear_predictions.append(linear_pred)

        # Hybrid: average both
        avg_pred = (lstm_pred + linear_pred) / 2
        hybrid_predictions.append(avg_pred)

        # Update inputs
        current_input_lstm = np.append(current_input_lstm[:, 1:, :],
                                       [[[lstm_pred]]], axis=1)
        current_input_linear = np.append(current_input_linear[1:],
                                         [linear_pred])

    return scaler.inverse_transform(np.array(
        hybrid_predictions).reshape(-1, 1))


def save_predictions_to_csv(dates, lstm, lin, hybrid, 
                            path='artifacts/predictions/future_preds.csv'):
    df = pd.DataFrame({
        'Date': dates,
        'LSTM': lstm.flatten(),
        'Linear': lin.flatten(),
        'Hybrid': hybrid.flatten()
    })
    df.to_csv(path, index=False)
