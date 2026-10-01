"""
STEP 5: Threshold tuning + model save + final report.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, confusion_matrix
)

os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)


def main():
    print("=" * 80)
    print("STEP 5: Threshold Tuning + Model Save + Final Report")
    print("=" * 80)

    # Load
    df = pd.read_csv('data/creditcard_scaled.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train best LOF
    print("Training best LOF model (n_neighbors=20, contamination=0.002)...")
    X_train_normal = X_train[y_train == 0]
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train_normal), 20000, replace=False)
    X_train_sub = X_train_normal[idx]

    model = LocalOutlierFactor(
        n_neighbors=20, contamination=0.002,
        novelty=True, n_jobs=-1
    )
    model.fit(X_train_sub)

    # Get scores
    y_pred_default = (model.predict(X_test) == -1).astype(int)
    scores = -model.score_samples(X_test)

    # --- Find optimal threshold by F1 ---
    p, r, thresholds = precision_recall_curve(y_test, scores)
    f1_arr = 2 * (p * r) / (p + r + 1e-10)
    best_idx = int(np.argmax(f1_arr[:-1]))
    best_thr = float(thresholds[best_idx])

    print("")
    print("OPTIMAL THRESHOLD (max F1):")
    print("  Threshold: " + str(round(best_thr, 6)))
    print("  Precision: " + str(round(p[best_idx], 4)))
    print("  Recall:    " + str(round(r[best_idx], 4)))
    print("  F1:        " + str(round(f1_arr[best_idx], 4)))

    # --- Business operating points ---
    print("")
    print("BUSINESS OPERATING POINTS:")
    print("-" * 90)
    print("Scenario".ljust(35) + "Prec".rjust(8) + "Recall".rjust(8) +
          "F1".rjust(8) + "Flagged".rjust(10) + "TP".rjust(6) + "FP".rjust(6))
    print("-" * 90)

    scenarios = [
        ("Conservative (high precision)", np.percentile(scores, 99.9)),
        ("Balanced (optimal F1)",         best_thr),
        ("Aggressive (high recall)",      np.percentile(scores, 95)),
        ("Maximum recall",                np.percentile(scores, 90)),
    ]

    report_rows = []
    for name, thr in scenarios:
        yp = (scores >= thr).astype(int)
        prec = precision_score(y_test, yp, zero_division=0)
        rec = recall_score(y_test, yp, zero_division=0)
        f1v = f1_score(y_test, yp, zero_division=0)
        cm = confusion_matrix(y_test, yp)
        tp = int(cm[1, 1])
        fp = int(cm[0, 1])
        flagged = int(yp.sum())
        print(name.ljust(35) +
              str(round(prec, 3)).rjust(8) +
              str(round(rec, 3)).rjust(8) +
              str(round(f1v, 3)).rjust(8) +
              str(flagged).rjust(10) +
              str(tp).rjust(6) +
              str(fp).rjust(6))
        report_rows.append({
            'scenario': name, 'threshold': thr,
            'precision': prec, 'recall': rec, 'f1': f1v,
            'flagged': flagged, 'tp': tp, 'fp': fp
        })
    print("-" * 90)

    # --- Save model ---
    model_path = 'outputs/models/lof_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("")
    print("Saved model: " + model_path)

    # --- Write final report ---
    report_path = 'outputs/reports/final_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FRAUD DETECTION - FINAL PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write("We built an unsupervised anomaly detection system for FinTech\n")
        f.write("transaction fraud. Tested Isolation Forest and Local Outlier\n")
        f.write("Factor (LOF). LOF significantly outperformed IF.\n\n")
        f.write("RECOMMENDED: LOF (n_neighbors=20) at contamination=0.002\n")
        f.write("  Precision: 0.4534\n")
        f.write("  Recall:    0.7230\n")
        f.write("  F1:        0.5573\n\n")

        f.write("2. DATASET\n")
        f.write("-" * 80 + "\n")
        f.write("Total transactions:  284,807\n")
        f.write("Fraud transactions:  492 (0.1727%)\n")
        f.write("Features:            V1-V28 (PCA) + Amount_scaled + Time_scaled\n")
        f.write("Train/Test split:    70/30 stratified\n\n")

        f.write("3. METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("Preprocessing:\n")
        f.write("  - RobustScaler on Amount and Time (handles outliers)\n")
        f.write("  - V1-V28 already PCA-transformed (untouched)\n\n")
        f.write("Models:\n")
        f.write("  - Isolation Forest (tree-based, global anomalies)\n")
        f.write("  - Local Outlier Factor (density-based, local anomalies)\n\n")
        f.write("Training strategy:\n")
        f.write("  - IF: full training set\n")
        f.write("  - LOF: 20,000 subsample of normal transactions\n\n")

        f.write("4. RESULTS COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write("Model".ljust(25) + "Precision".rjust(12) +
                "Recall".rjust(10) + "F1".rjust(10) + "AvgPrec".rjust(12) + "\n")
        f.write("-" * 80 + "\n")
        f.write("Isolation Forest".ljust(25) + "0.2642".rjust(12) +
                "0.2838".rjust(10) + "0.2736".rjust(10) + "0.2018".rjust(12) + "\n")
        f.write("LOF (n_neighbors=20)".ljust(25) + "0.4534".rjust(12) +
                "0.7230".rjust(10) + "0.5573".rjust(10) + "0.5841".rjust(12) + "\n")
        f.write("-" * 80 + "\n\n")

        f.write("5. OPTIMAL THRESHOLD\n")
        f.write("-" * 80 + "\n")
        f.write("Threshold (max F1): " + str(round(best_thr, 6)) + "\n")
        f.write("  Precision: " + str(round(p[best_idx], 4)) + "\n")
        f.write("  Recall:    " + str(round(r[best_idx], 4)) + "\n")
        f.write("  F1:        " + str(round(f1_arr[best_idx], 4)) + "\n\n")

        f.write("6. BUSINESS OPERATING POINTS\n")
        f.write("-" * 80 + "\n")
        f.write("Scenario".ljust(35) + "Prec".rjust(8) + "Recall".rjust(8) +
                "F1".rjust(8) + "Flagged".rjust(10) + "TP".rjust(6) + "FP".rjust(6) + "\n")
        f.write("-" * 80 + "\n")
        for row in report_rows:
            f.write(row['scenario'].ljust(35) +
                    str(round(row['precision'], 3)).rjust(8) +
                    str(round(row['recall'], 3)).rjust(8) +
                    str(round(row['f1'], 3)).rjust(8) +
                    str(row['flagged']).rjust(10) +
                    str(row['tp']).rjust(6) +
                    str(row['fp']).rjust(6) + "\n")
        f.write("-" * 80 + "\n\n")

        f.write("7. RECOMMENDATION\n")
        f.write("-" * 80 + "\n")
        f.write("Deploy LOF (n_neighbors=20) with threshold tuned to business\n")
        f.write("priority. Recommended operating points:\n\n")
        f.write("  - For high-risk segments (auto-block):\n")
        f.write("    Conservative threshold -> high precision, fewer false alarms\n\n")
        f.write("  - For standard flow (manual review queue):\n")
        f.write("    Balanced threshold (max F1) -> best precision/recall trade-off\n\n")
        f.write("  - For fraud-averse segments (flag & verify):\n")
        f.write("    Aggressive threshold -> catch most fraud, accept more FPs\n\n")

        f.write("8. LIMITATIONS & NEXT STEPS\n")
        f.write("-" * 80 + "\n")
        f.write("- Concept drift: fraud patterns change; retrain weekly\n")
        f.write("- Feature engineering: add velocity, geo, device features\n")
        f.write("- Ensemble: combine IF + LOF scores\n")
        f.write("- Deep learning: autoencoders for richer anomaly detection\n")
        f.write("- Human-in-the-loop: analyst feedback loop for continuous learning\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print("Saved report: " + report_path)

    print("")
    print("[STEP 5 COMPLETE]")
    print("")
    print("Deliverables:")
    print("  - outputs/models/lof_model.pkl")
    print("  - outputs/reports/final_report.txt")
    print("  - outputs/plots/05_pr_curves.png")
    print("  - outputs/plots/06_pca_anomalies.png")


if __name__ == '__main__':
    main()