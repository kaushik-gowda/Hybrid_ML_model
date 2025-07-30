import numpy as np


def forecast_with_linear(model, recent_data, n_future, scaler):
    """
    Forecast future values using a trained Linear Regression model.
    
    Args:
        model: Trained linear model.
        recent_data: Most recent input sequence (scaled).
        n_future: Number of time steps to forecast.
        scaler: Scaler used for inverse transformation.

    Returns:
        Array of forecasted values (inverse transformed).
    """
    predictions = []
    current_input = recent_data.copy()

    for _ in range(n_future):
        input_reshaped = current_input.reshape(1, -1)
        prediction = model.predict(input_reshaped)[0]
        predictions.append(prediction)
        current_input = np.append(current_input[1:], prediction)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
