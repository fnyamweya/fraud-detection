import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_random_forest_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30]
    }

    logging.info("Starting GridSearch for Random Forest...")
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    logging.info(f"GridSearch complete. Best parameters: {grid_search.best_params_}")
    
    # Save the model
    joblib.dump(model, 'models/random_forest_model.pkl')
    logging.info("Random Forest model saved to models/random_forest_model.pkl")
    
    return model

def evaluate_random_forest_model(model, X_test, y_test):
    logging.info("Evaluating Random Forest model...")
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred)
    logging.info("Random Forest evaluation complete.")
    
    return y_pred, metrics
