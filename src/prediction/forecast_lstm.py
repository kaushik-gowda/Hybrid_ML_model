import numpy as np


def forecast_with_lstm(model, recent_data, n_future, scaler):
    """
    Forecast future values using the trained LSTM model.

    Args:
        model: Trained LSTM model.
        recent_data: Most recent input sequence (scaled).
        n_future: Number of time steps to forecast.
        scaler: Scaler used for inverse transformation.

    Returns:
        Array of forecasted values (inverse transformed).
    """
    predictions = []
    current_input = recent_data.copy()

    for _ in range(n_future):
        prediction = model.predict(current_input.reshape(
            1,
            current_input.shape[0],
            current_input.shape[1]), verbose=0)
        predictions.append(prediction[0, 0])
        current_input = np.append(current_input[:, 1:, :], [[prediction]],
                                  axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
