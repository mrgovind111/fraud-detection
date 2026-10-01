"""
Experiment: try different contamination values for Isolation Forest.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import train_isolation_forest, predict_isolation_forest
from evaluation import evaluate


def main():
    df = pd.read_csv('data/creditcard_scaled.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_normal = X_train[y_train == 0]

    print("")
    print("Testing contamination values...")
    print("=" * 70)
    print("Contamination   Precision   Recall    F1      AvgPrec")
    print("=" * 70)

    results = []
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
        model = train_isolation_forest(
            X_train_normal, contamination=c, n_estimators=100
        )
        y_pred, scores = predict_isolation_forest(model, X_test)
        r = evaluate(y_test, y_pred, scores, name="IF(c=" + str(c) + ")")
        results.append((c, r))

    print("")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Contam      Prec     Recall   F1       AvgPrec")
    print("-" * 70)
    for c, r in results:
        print(str(c).ljust(10) +
              str(round(r['precision'], 4)).ljust(9) +
              str(round(r['recall'], 4)).ljust(9) +
              str(round(r['f1'], 4)).ljust(9) +
              str(round(r['avg_precision'], 4)).ljust(9))
    print("=" * 70)


if __name__ == '__main__':
    main()