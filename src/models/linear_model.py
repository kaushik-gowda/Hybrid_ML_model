from sklearn.linear_model import LinearRegression
import joblib


def train_linear_model(X_train, y_train):
    """Train a simple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def save_linear_model(model, path='artifacts/models/linear_model.pkl'):
    joblib.dump(model, path)


def save_scaler(scaler, path='artifacts/scalers/scaler.pkl'):
    joblib.dump(scaler, path)
