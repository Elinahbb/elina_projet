"""
Model evaluation and visualization.
"""

from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and print results."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    return acc
