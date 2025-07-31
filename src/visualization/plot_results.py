import matplotlib.pyplot as plt
import os
import logging  # make sure your logger is initialized before calling this


def plot_predictions(dates, lstm, lin, hybrid, 
                     save_path='artifacts/plots/future_plot.png'):
    try:
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(dates, lstm, label='LSTM')
        plt.plot(dates, lin, label='Linear')
        plt.plot(dates, hybrid, label='Hybrid')
        plt.title('Future Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save and close
        plt.savefig(save_path)
        plt.close()

        logging.info(f"Plot successfully saved to {save_path}")

    except Exception as e:
        logging.error(f"Error while plotting predictions: {str(e)}")
        raise e
