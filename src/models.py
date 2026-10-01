"""
Model wrappers: Isolation Forest and Local Outlier Factor.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def train_isolation_forest(X_train_normal,
                           contamination=0.001,
                           n_estimators=100,
                           random_state=42):
    """
    Train Isolation Forest on NORMAL transactions only.
    Returns fitted model.
    """
    print("Training Isolation Forest on " +
          str(len(X_train_normal)) + " normal transactions...")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_normal)
    print("  Done. n_estimators=" + str(n_estimators) +
          " contamination=" + str(contamination))
    return model


def predict_isolation_forest(model, X_test):
    """Return (y_pred, scores) where 1 = anomaly and higher score = more anomalous."""
    raw = model.predict(X_test)
    y_pred = (raw == -1).astype(int)
    scores = -model.score_samples(X_test)
    return y_pred, scores


def train_lof(X_train_normal,
              contamination=0.001,
              n_neighbors=20,
              random_state=42):
    """
    Train Local Outlier Factor. novelty=True allows predicting on new data.
    Train on NORMAL transactions only.
    """
    print("Training LOF on " + str(len(X_train_normal)) +
          " normal transactions (this may take a few minutes)...")
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1
    )
    model.fit(X_train_normal)
    print("  Done. n_neighbors=" + str(n_neighbors) +
          " contamination=" + str(contamination))
    return model


def predict_lof(model, X_test):
    raw = model.predict(X_test)
    y_pred = (raw == -1).astype(int)
    scores = -model.score_samples(X_test)
    return y_pred, scores