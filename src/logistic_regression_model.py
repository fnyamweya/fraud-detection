import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_logistic_regression_model(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }

    logging.info("Starting GridSearch for Logistic Regression...")
    
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    logging.info(f"GridSearch complete. Best parameters: {grid_search.best_params_}")
    
    # Save the model
    joblib.dump(model, 'models/logistic_regression_model.pkl')
    logging.info("Logistic Regression model saved to models/logistic_regression_model.pkl")
    
    return model

def evaluate_logistic_regression_model(model, X_test, y_test):
    logging.info("Evaluating Logistic Regression model...")
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred)
    logging.info("Logistic Regression evaluation complete.")
    
    return y_pred, metrics
