import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_test, y_pred):
    """
    Plots the confusion matrix using the true labels and predictions.

    Args:
        y_test (array-like): True labels of the test data.
        y_pred (array-like): Predicted labels from the model.

    Returns:
        None: Saves the confusion matrix plot as 'confusion_matrix.png' and displays it.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance of a model. Primarily works with tree-based models like RandomForest.

    Args:
        model: Trained model (should have a feature_importances_ attribute).
        feature_names: List of feature names for labeling the importance plot.

    Returns:
        None: Saves the feature importance plot as 'feature_importance.png' and displays it.
    """
    importances = model.feature_importances_
    indices = importances.argsort()

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.savefig('outputs/feature_importance.png')
    plt.show()

def plot_shap_summary(model, X_test):
    """
    Plots a SHAP summary plot to interpret feature contributions to predictions.

    Args:
        model: Trained model (should be tree-based, such as RandomForest or XGBoost).
        X_test: Test data used for generating SHAP values.

    Returns:
        None: Displays SHAP summary plot.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

def evaluate_model_performance(y_test, y_pred):
    """
    Evaluates the model performance using common metrics such as accuracy, precision, recall, and F1-score.

    Args:
        y_test (array-like): True labels of the test data.
        y_pred (array-like): Predicted labels from the model.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    performance = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    return performance
