"""
This module defines and trains the machine learning models
used to predict the daily direction of Bitcoin prices.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Logistic Regression is used as a simple baseline model for binary classification (price up or down)

def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(
        max_iter=500,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# Random Forest is an ensemble model that can capture non-linear relationships between features

def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# K-Nearest Neighbors (KNN) predicts the class based on the most similar observations in the training set

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model
