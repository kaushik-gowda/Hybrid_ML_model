import numpy as np


def forecast_lstm(model, last_sequence, scaler, steps=10):
    predictions = []
    sequence = last_sequence.copy()
    for _ in range(steps):
        pred = model.predict(sequence)[0, 0]
        predictions.append(pred)
        pred_reshaped = np.array([[pred]]).reshape(1, 1, 1)
        sequence = np.append(sequence[:, 1:, :], pred_reshaped, axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
