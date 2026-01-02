"""
Model definitions and training.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(
        max_iter=500,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model
