import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.logger import logging


def train_linear_model(X_train, y_train, X_val=None, y_val=None,
                       model_name="linear_model.pkl"):
    logging.info("Training Linear Regression model...")

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Optional: Evaluate the model
    if X_val is not None and y_val is not None:
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        logging.info(f"Validation MSE: {mse}")

    # Save model to artifacts folder
    artifacts_dir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, model_name)
    joblib.dump(model, model_path)
    logging.info(f"Linear model saved to {model_path}")

    return model
