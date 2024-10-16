import pandas as pd
import numpy as np

def add_transaction_features(data):
    """
    Adds features related to transaction amounts.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with new transaction-related features.
    """
    # Replace 'TransactionAmt' with 'Amount'
    data['Amount_to_Mean'] = data['Amount'] / data['Amount'].mean()
    data['Amount_to_Std'] = data['Amount'] / data['Amount'].std()

    # Log-transformed Transaction Amount
    data['Amount_log'] = np.log1p(data['Amount'])

    print("Transaction features added.")
    return data

def add_time_features(data):
    """
    Adds time-related features from the transaction timestamp.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with new time-related features.
    """
    # Convert 'Time' to datetime if necessary
    data['Time'] = pd.to_timedelta(data['Time'], unit='s')

    # Extract useful time-related features
    data['hour'] = data['Time'].dt.seconds // 3600
    data['day'] = data['Time'].dt.days

    print("Time features added.")
    return data

def add_derived_features(data):
    """
    Adds additional derived features based on domain knowledge.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with new derived features.
    """
    # Example derived feature: Ratio of transaction amount to total amount
    if 'total_transaction_amt' in data.columns:
        data['Amount_to_Total'] = data['Amount'] / data['total_transaction_amt']

    print("Derived features added.")
    return data

def apply_feature_engineering(data):
    """
    Applies all feature engineering steps to the dataset.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with all engineered features.
    """
    # Add transaction-related features
    data = add_transaction_features(data)

    # Add time-related features
    data = add_time_features(data)

    # Add derived features based on domain knowledge
    data = add_derived_features(data)

    print("Feature engineering completed.")
    return data
