import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Loads the dataset from a CSV file.

    Args:
        file_path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(data):
    """
    Handles missing values in the dataset by filling them with the mean of the column.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with missing values handled.
    """
    # Fill missing values with column means
    data.fillna(data.mean(), inplace=True)
    return data

def preprocess_data(data):
    """
    Preprocesses the data by handling missing values, splitting the data, and scaling features.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        X_train, X_test, y_train, y_test (np.array): Split and preprocessed features and labels.
    """
    # Drop 'Time' column because it has timedelta type
    data = data.drop(columns=['Time'])

    # Assume 'Class' is the target variable, where 1 = Fraud, 0 = No Fraud
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Handle missing values
    X = handle_missing_values(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test
