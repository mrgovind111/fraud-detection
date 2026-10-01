"""
STEP 2: Run Isolation Forest pipeline (multi-file version).
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow imports from same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import train_isolation_forest, predict_isolation_forest
from evaluation import (
    evaluate, plot_confusion_matrix, plot_score_distribution
)


def load_scaled_data(path='data/creditcard_scaled.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Run src/preprocessing.py first to create " + path
        )
    df = pd.read_csv(path)
    print("Loaded: " + path + "  shape=" + str(df.shape))
    return df


def split_data(df, test_size=0.3, random_state=42):
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    print("")
    print("Train size: " + str(len(X_train)) +
          "  (fraud=" + str(int(y_train.sum())) + ")")
    print("Test  size: " + str(len(X_test)) +
          "  (fraud=" + str(int(y_test.sum())) + ")")
    return X_train, X_test, y_train, y_test


def main():
    print("=" * 60)
    print("STEP 2: Isolation Forest Fraud Detection")
    print("=" * 60)

    df = load_scaled_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Train ONLY on normal transactions
    X_train_normal = X_train[y_train == 0]

    model = train_isolation_forest(
        X_train_normal,
        contamination=0.001,
        n_estimators=100
    )

    y_pred, scores = predict_isolation_forest(model, X_test)

    results = evaluate(y_test, y_pred, scores, name="Isolation Forest")

    plot_confusion_matrix(results['cm'], "Isolation Forest")
    plot_score_distribution(scores, y_test, "Isolation Forest")

    print("")
    print("[STEP 2 COMPLETE]")


if __name__ == '__main__':
    main()