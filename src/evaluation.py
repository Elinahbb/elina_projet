"""
Model evaluation and visualization.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a classification model and return key performance metrics.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall
    }
