import matplotlib.pyplot as plt


def plot_predictions(dates, lstm, lin, hybrid, 
                     save_path='artifacts/plots/future_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, lstm, label='LSTM')
    plt.plot(dates, lin, label='Linear')
    plt.plot(dates, hybrid, label='Hybrid')
    plt.title('Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
