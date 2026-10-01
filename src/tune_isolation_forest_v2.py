"""
Experiment v2: compare training strategies and hyperparameters.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import predict_isolation_forest


def run_one(X_train, X_test, y_test, name, **kwargs):
    model = IsolationForest(random_state=42, n_jobs=-1, **kwargs)
    model.fit(X_train)
    y_pred, scores = predict_isolation_forest(model, X_test)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(name.ljust(58) +
          "P=" + str(round(p, 3)).ljust(9) +
          "R=" + str(round(r, 3)).ljust(9) +
          "F1=" + str(round(f1, 3)).ljust(9))
    return {'name': name, 'p': p, 'r': r, 'f1': f1}


def main():
    df = pd.read_csv('data/creditcard_scaled.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train_normal = X_train[y_train == 0]

    print("")
    print("=" * 110)
    print("EXPERIMENT v2: training strategy + hyperparameter sweep")
    print("=" * 110)
    print("Setup".ljust(58) + "Precision  Recall   F1")
    print("-" * 110)

    experiments = [
        # name, X_train, contamination, n_estimators, max_samples
        ("normal-only  c=0.002 n=100 auto",  X_train_normal, 0.002, 100, 'auto'),
        ("normal-only  c=0.005 n=100 auto",  X_train_normal, 0.005, 100, 'auto'),
        ("FULL train   c=0.002 n=100 auto",  X_train, 0.002, 100, 'auto'),
        ("FULL train   c=0.005 n=100 auto",  X_train, 0.005, 100, 'auto'),
        ("FULL train   c=0.01  n=100 auto",  X_train, 0.01, 100, 'auto'),
        ("FULL train   c=0.01  n=100 ms=256", X_train, 0.01, 100, 256),
        ("FULL train   c=0.01  n=200 ms=256", X_train, 0.01, 200, 256),
        ("FULL train   c=0.01  n=300 ms=512", X_train, 0.01, 300, 512),
        ("FULL train   c=0.02  n=100 ms=256", X_train, 0.02, 100, 256),
        ("FULL train   c=0.005 n=200 ms=256", X_train, 0.005, 200, 256),
        ("FULL train   c=0.005 n=300 ms=512", X_train, 0.005, 300, 512),
        ("FULL train   c=0.002 n=200 ms=512", X_train, 0.002, 200, 512),
    ]

    results = []
    for name, Xtr, c, n, ms in experiments:
        r = run_one(
            Xtr, X_test, y_test, name,
            contamination=c,
            n_estimators=n,
            max_samples=ms
        )
        results.append(r)

    best = max(results, key=lambda r: r['f1'])
    print("")
    print("=" * 110)
    print("BEST BY F1: " + best['name'])
    print("  Precision=" + str(round(best['p'], 4)) +
          "  Recall=" + str(round(best['r'], 4)) +
          "  F1=" + str(round(best['f1'], 4)))
    print("=" * 110)


if __name__ == '__main__':
    main()