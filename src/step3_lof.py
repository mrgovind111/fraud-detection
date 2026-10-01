"""
STEP 3: Local Outlier Factor for fraud detection.
"""
import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_lof(X_train, X_test, y_test, name, **kwargs):
    t0 = time.time()
    model = LocalOutlierFactor(novelty=True, n_jobs=-1, **kwargs)
    model.fit(X_train)
    fit_time = time.time() - t0

    raw = model.predict(X_test)
    y_pred = (raw == -1).astype(int)
    scores = -model.score_samples(X_test)

    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(name.ljust(45) +
          "P=" + str(round(p, 3)).ljust(9) +
          "R=" + str(round(r, 3)).ljust(9) +
          "F1=" + str(round(f1, 3)).ljust(9) +
          "t=" + str(round(fit_time, 1)) + "s")
    return {'name': name, 'p': p, 'r': r, 'f1': f1,
            'fit_time': fit_time, 'scores': scores, 'y_pred': y_pred}


def main():
    df = pd.read_csv('data/creditcard_scaled.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("")
    print("=" * 100)
    print("STEP 3: Local Outlier Factor (LOF) sweep")
    print("=" * 100)

    # LOF is expensive on large data — subsample train set for speed
    # (use only normal + some fraud, cap at 20,000 for reasonable runtime)
    X_train_normal = X_train[y_train == 0]
    n_sub = min(20000, len(X_train_normal))
    import numpy as np
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train_normal), n_sub, replace=False)
    X_train_sub = X_train_normal[idx]
    print("Subsampled train set: " + str(n_sub) + " normal transactions")
    print("")

    print("Setup".ljust(45) + "Precision  Recall   F1       Time")
    print("-" * 100)

    experiments = [
        ("LOF n_neighbors=10 c=0.002", 10, 0.002),
        ("LOF n_neighbors=20 c=0.002", 20, 0.002),
        ("LOF n_neighbors=20 c=0.005", 20, 0.005),
        ("LOF n_neighbors=20 c=0.01",  20, 0.01),
        ("LOF n_neighbors=50 c=0.005", 50, 0.005),
        ("LOF n_neighbors=50 c=0.01",  50, 0.01),
        ("LOF n_neighbors=100 c=0.005", 100, 0.005),
        ("LOF n_neighbors=100 c=0.01",  100, 0.01),
    ]

    results = []
    for name, k, c in experiments:
        r = run_lof(
            X_train_sub, X_test, y_test, name,
            n_neighbors=k, contamination=c
        )
        results.append(r)

    best = max(results, key=lambda r: r['f1'])
    print("")
    print("=" * 100)
    print("BEST LOF BY F1: " + best['name'])
    print("  Precision=" + str(round(best['p'], 4)) +
          "  Recall=" + str(round(best['r'], 4)) +
          "  F1=" + str(round(best['f1'], 4)))
    print("=" * 100)


if __name__ == '__main__':
    main()