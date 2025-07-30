from sklearn.linear_model import LinearRegression
import joblib


def build_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def save_linear_model(model, path='artifacts/models/linear_model.pkl'):
    joblib.dump(model, path)


def save_scaler(scaler, path='artifacts/scalers/scaler.pkl'):
    joblib.dump(scaler, path)
