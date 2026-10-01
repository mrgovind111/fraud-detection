"""
STEP 4: Final comparison of Isolation Forest vs LOF.
Generates:
  - Comparison table
  - Precision-Recall curves
  - 2D PCA visualization
  - Optimal threshold recommendation
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)


def train_if(X_train, contamination=0.002, n_estimators=100):
    t0 = time.time()
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    fit_time = time.time() - t0
    return model, fit_time


def train_lof_model(X_train, contamination=0.002, n_neighbors=20):
    t0 = time.time()
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1
    )
    model.fit(X_train)
    fit_time = time.time() - t0
    return model, fit_time


def evaluate_model(y_true, y_pred, scores, name, fit_time):
    return {
        'name': name,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'ap': average_precision_score(y_true, scores),
        'fit_time': fit_time,
        'y_true': y_true,
        'y_pred': y_pred,
        'scores': scores,
    }


def plot_pr_curves(results_list, save_path):
    plt.figure(figsize=(9, 6))
    for r in results_list:
        p, rec, _ = precision_recall_curve(r['y_true'], r['scores'])
        plt.plot(rec, p, linewidth=2,
                 label=r['name'] + " (AP=" + str(round(r['ap'], 3)) + ")")
    plt.axhline(y=0.0017, color='gray', linestyle='--',
                label='Random baseline (0.17%)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Isolation Forest vs LOF')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Saved: " + save_path)


def plot_pca_anomalies(X, y_true, results_list, save_path):
    """3-panel: ground truth, IF anomalies, LOF anomalies."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel 1: ground truth
    axes[0].scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1],
                    c='steelblue', s=3, alpha=0.4, label='Normal')
    axes[0].scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1],
                    c='red', s=20, alpha=0.9, label='Fraud (true)')
    axes[0].set_title('Ground Truth (PCA)')
    axes[0].legend()

    # Panel 2 & 3: model predictions
    for i, r in enumerate(results_list):
        ax = axes[i + 1]
        normal_mask = r['y_pred'] == 0
        anomaly_mask = r['y_pred'] == 1
        ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                   c='steelblue', s=3, alpha=0.4, label='Pred Normal')
        ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
                   c='crimson', s=20, alpha=0.9, label='Pred Anomaly')
        ax.set_title(r['name'] + ' (P=' +
                     str(round(r['precision'], 3)) + ', R=' +
                     str(round(r['recall'], 3)) + ')')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Saved: " + save_path)


def find_optimal_threshold(y_true, scores):
    """Find threshold maximizing F1."""
    p, r, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * (p * r) / (p + r + 1e-10)
    best_idx = np.argmax(f1[:-1])
    return {
        'threshold': thresholds[best_idx],
        'precision': p[best_idx],
        'recall': r[best_idx],
        'f1': f1[best_idx],
    }


def main():
    print("=" * 90)
    print("STEP 4: Final Comparison — Isolation Forest vs Local Outlier Factor")
    print("=" * 90)

    # Load data
    df = pd.read_csv('data/creditcard_scaled.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Train: " + str(len(X_train)) + " (fraud=" + str(int(y_train.sum())) + ")")
    print("Test:  " + str(len(X_test)) + " (fraud=" + str(int(y_test.sum())) + ")")

    # Train Isolation Forest on FULL train set (best config from v2)
    print("")
    print("Training Isolation Forest...")
    if_model, if_time = train_if(X_train, contamination=0.002, n_estimators=100)
    if_pred = (if_model.predict(X_test) == -1).astype(int)
    if_scores = -if_model.score_samples(X_test)
    if_results = evaluate_model(y_test, if_pred, if_scores,
                                "Isolation Forest", if_time)

    # Train LOF on subsample of normal transactions (best config)
    print("Training LOF...")
    X_train_normal = X_train[y_train == 0]
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train_normal), 20000, replace=False)
    X_train_sub = X_train_normal[idx]
    lof_model, lof_time = train_lof_model(
        X_train_sub, contamination=0.002, n_neighbors=20
    )
    lof_pred = (lof_model.predict(X_test) == -1).astype(int)
    lof_scores = -lof_model.score_samples(X_test)
    lof_results = evaluate_model(y_test, lof_pred, lof_scores,
                                 "LOF (n=20)", lof_time)

    # Comparison table
    print("")
    print("=" * 90)
    print("COMPARISON")
    print("=" * 90)
    print("Model".ljust(25) + "Precision".rjust(12) + "Recall".rjust(10) +
          "F1".rjust(10) + "AvgPrec".rjust(12) + "Time(s)".rjust(12))
    print("-" * 90)
    for r in [if_results, lof_results]:
        print(r['name'].ljust(25) +
              str(round(r['precision'], 4)).rjust(12) +
              str(round(r['recall'], 4)).rjust(10) +
              str(round(r['f1'], 4)).rjust(10) +
              str(round(r['ap'], 4)).rjust(12) +
              str(round(r['fit_time'], 2)).rjust(12))
    print("=" * 90)

    # Optimal thresholds
    print("")
    print("OPTIMAL THRESHOLDS (max F1):")
    print("-" * 90)
    for r in [if_results, lof_results]:
        opt = find_optimal_threshold(r['y_true'], r['scores'])
        print(r['name'].ljust(25) +
              "thr=" + str(round(opt['threshold'], 4)).ljust(10) +
              "P=" + str(round(opt['precision'], 3)).ljust(8) +
              "R=" + str(round(opt['recall'], 3)).ljust(8) +
              "F1=" + str(round(opt['f1'], 3)))
    print("=" * 90)

    # Plot PR curves
    plot_pr_curves([if_results, lof_results],
                   'outputs/plots/05_pr_curves.png')

    # Plot PCA visualization
    plot_pca_anomalies(X_test, y_test, [if_results, lof_results],
                       'outputs/plots/06_pca_anomalies.png')

    # Write report
    with open('outputs/reports/comparison.txt', 'w') as f:
        f.write("FRAUD DETECTION — MODEL COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write("Dataset: 284,807 transactions, 492 fraud (0.17%)\n")
        f.write("Train: " + str(len(X_train)) + ", Test: " + str(len(X_test)) + "\n\n")
        f.write("RESULTS\n")
        f.write("-" * 60 + "\n")
        for r in [if_results, lof_results]:
            f.write(r['name'] + "\n")
            f.write("  Precision: " + str(round(r['precision'], 4)) + "\n")
            f.write("  Recall:    " + str(round(r['recall'], 4)) + "\n")
            f.write("  F1:        " + str(round(r['f1'], 4)) + "\n")
            f.write("  Avg Prec:  " + str(round(r['ap'], 4)) + "\n")
            f.write("  Fit time:  " + str(round(r['fit_time'], 2)) + "s\n\n")
        best = lof_results if lof_results['f1'] > if_results['f1'] else if_results
        f.write("RECOMMENDATION\n")
        f.write("-" * 60 + "\n")
        f.write("Best model by F1: " + best['name'] + "\n")
        f.write("  Precision = " + str(round(best['precision'], 4)) + "\n")
        f.write("  Recall    = " + str(round(best['recall'], 4)) + "\n")
        f.write("  F1        = " + str(round(best['f1'], 4)) + "\n")
    print("Saved: outputs/reports/comparison.txt")

    print("")
    print("[STEP 4 COMPLETE]")


if __name__ == '__main__':
    main()