import numpy as np
import pandas as pd


def forecast_linear(model, recent_data, scaler, steps=10):
    feature_names = ['Lag_1', 'Lag_2', 'Lag_3']
    predictions = []
    data = recent_data.copy()
    for _ in range(steps):
        input_df = pd.DataFrame([data], columns=feature_names)
        pred = model.predict(input_df)[0]
        predictions.append(pred)
        data = np.append(data[1:], pred)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
