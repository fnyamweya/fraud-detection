from sklearn.metrics import classification_report
from .logistic_regression_model import build_logistic_regression_model, evaluate_logistic_regression_model
from .random_forest_model import build_random_forest_model, evaluate_random_forest_model

def build_model(model_type, X_train, y_train):
    if model_type == 'logistic_regression':
        return build_logistic_regression_model(X_train, y_train)
    elif model_type == 'random_forest':
        return build_random_forest_model(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def evaluate_model(model_type, model, X_test, y_test):
    if model_type == 'logistic_regression':
        return evaluate_logistic_regression_model(model, X_test, y_test)
    elif model_type == 'random_forest':
        return evaluate_random_forest_model(model, X_test, y_test)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
