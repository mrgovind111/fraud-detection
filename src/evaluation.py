"""
Evaluation utilities: metrics, confusion matrix, score plots.
"""
import os
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(y_true, y_pred, scores, name="Model"):
    """Compute precision, recall, F1, confusion matrix."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)

    print("")
    print("=" * 60)
    print("RESULTS - " + name)
    print("=" * 60)
    print("Precision:       " + str(round(precision, 4)))
    print("Recall:          " + str(round(recall, 4)))
    print("F1-Score:        " + str(round(f1, 4)))
    print("Avg Precision:   " + str(round(ap, 4)))
    print("")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("                 Pred Normal   Pred Fraud")
    print("  Actual Normal  " + str(cm[0, 0]).rjust(10) +
          "  " + str(cm[0, 1]).rjust(10))
    print("  Actual Fraud   " + str(cm[1, 0]).rjust(10) +
          "  " + str(cm[1, 1]).rjust(10))
    print("")
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Fraud'],
                                zero_division=0))
    print("=" * 60)

    return {
        'name': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_precision': ap,
        'cm': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'scores': scores
    }


def plot_confusion_matrix(cm, name, save_path=None):
    os.makedirs('outputs/plots', exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix - ' + name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    if save_path is None:
        save_path = 'outputs/plots/cm_' + name.lower().replace(' ', '_') + '.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Saved: " + save_path)


def plot_score_distribution(scores, y_true, name, save_path=None):
    os.makedirs('outputs/plots', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(scores[y_true == 0], bins=80, alpha=0.6,
             label='Normal', color='steelblue', density=True)
    plt.hist(scores[y_true == 1], bins=80, alpha=0.7,
             label='Fraud', color='crimson', density=True)
    plt.xlabel('Anomaly Score (higher = more anomalous)')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution - ' + name)
    plt.legend()
    plt.tight_layout()
    if save_path is None:
        save_path = ('outputs/plots/score_dist_' +
                     name.lower().replace(' ', '_') + '.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Saved: " + save_path)


def print_comparison(results_list):
    """Print a side-by-side comparison table."""
    print("")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    header = "Model".ljust(20) + "Precision".rjust(12) + \
             "Recall".rjust(12) + "F1".rjust(12) + "Avg Prec".rjust(12)
    print(header)
    print("-" * 80)
    for r in results_list:
        row = r['name'].ljust(20) + \
              str(round(r['precision'], 4)).rjust(12) + \
              str(round(r['recall'], 4)).rjust(12) + \
              str(round(r['f1'], 4)).rjust(12) + \
              str(round(r['avg_precision'], 4)).rjust(12)
        print(row)
    print("=" * 80)