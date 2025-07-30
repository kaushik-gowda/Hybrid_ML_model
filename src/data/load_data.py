import pandas as pd


def load_stock_data(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)
    data = data[['Close']].sort_index()
    return data
