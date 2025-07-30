import pandas as pd


def forecast_hybrid(lstm_preds, lin_preds, alpha=0.7):
    return (alpha * lstm_preds) + ((1 - alpha) * lin_preds)


def save_predictions_to_csv(dates, lstm, lin, hybrid, 
                            path='artifacts/predictions/future_preds.csv'):
    df = pd.DataFrame({
        'Date': dates,
        'LSTM': lstm.flatten(),
        'Linear': lin.flatten(),
        'Hybrid': hybrid.flatten()
    })
    df.to_csv(path, index=False)
