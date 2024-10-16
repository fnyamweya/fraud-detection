from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import apply_feature_engineering
from src.random_forest_model import build_random_forest_model, evaluate_random_forest_model
from src.logistic_regression_model import build_logistic_regression_model, evaluate_logistic_regression_model
from src.utils import plot_confusion_matrix, evaluate_model_performance
import argparse
import logging
import os

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Argument parsing for flexibility
    parser = argparse.ArgumentParser(description="Fraud Detection System")
    parser.add_argument('--data', type=str, default='data/creditcard.csv', help='Path to the data file')
    args = parser.parse_args()

    # Check if the dataset exists
    if not os.path.exists(args.data):
        logging.error(f"Data file {args.data} not found.")
        return

    # Load and preprocess data
    logging.info("Loading data...")
    data = load_data(args.data)
    if data is None:
        logging.error("Failed to load data.")
        return
    
    # Apply feature engineering
    logging.info("Applying feature engineering...")
    data = apply_feature_engineering(data)

    # Preprocess data (splitting and scaling)
    logging.info("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Ask user to choose the model
    print("Choose a model to train:")
    print("1. Random Forest")
    print("2. Logistic Regression")
    
    model_choice = input("Enter 1 or 2: ")

    if model_choice == "1":
        logging.info("Building and training Random Forest model...")
        model = build_random_forest_model(X_train, y_train)
        y_pred, metrics = evaluate_random_forest_model(model, X_test, y_test)
    elif model_choice == "2":
        logging.info("Building and training Logistic Regression model...")
        model = build_logistic_regression_model(X_train, y_train)
        y_pred, metrics = evaluate_logistic_regression_model(model, X_test, y_test)
    else:
        logging.error("Invalid choice. Please enter 1 or 2.")
        return

    # Display metrics
    print(metrics)
    performance = evaluate_model_performance(y_test, y_pred)
    logging.info(f"Model performance: {performance}")

    # Plot confusion matrix
    logging.info("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    logging.info("Fraud detection process completed.")

if __name__ == '__main__':
    main()
