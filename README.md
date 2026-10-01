# Real-Time Anomaly Detection in Financial Transactions

An end-to-end **unsupervised machine learning system** for detecting fraudulent FinTech transactions using **Isolation Forest** and **Local Outlier Factor (LOF)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Problem

Financial fraud is a **multi-billion dollar problem**. In a FinTech like SecurePay, distinguishing a legitimate purchase from a fraudulent one in milliseconds is the difference between profit and reputational ruin.

**The challenge:** Fraud is statistically rare — only **0.17%** of transactions in our dataset. Traditional supervised classifiers fail because they achieve 99.83% accuracy by simply predicting "Legitimate" every time.

**The solution:** Build an **unsupervised anomaly detection system** that learns "what normal looks like" and flags deviations — without ever being trained on labeled fraud.

---

## 📊 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.1727%)
- **Features:** V1–V28 (PCA-transformed), Amount, Time

---

## 🔬 Methodology

### Phase 1 — Data Preprocessing
- Exploratory Data Analysis (amount/time distributions, class imbalance)
- **RobustScaler** applied to `Amount` and `Time` (handles outliers using IQR)
- PCA features (V1–V28) left untouched — already scaled

### Phase 2 — Unsupervised Models

| Algorithm | Type | Strength |
|-----------|------|----------|
| **Isolation Forest** | Tree-based | Global anomalies |
| **Local Outlier Factor** | Density-based | Local anomalies |

Both trained on **normal transactions only** to establish a baseline of acceptable behavior.

### Phase 3 — Evaluation & Threshold Tuning
- **Precision, Recall, F1** — never accuracy (imbalanced data)
- **Precision-Recall curve** — visualize trade-off
- **Business operating points** — conservative, balanced, aggressive

---

## 🏆 Results

| Model | Precision | Recall | F1 | Avg Precision |
|-------|-----------|--------|-----|---------------|
| Isolation Forest | 0.264 | 0.284 | 0.274 | 0.202 |
| **LOF (n=20)** | **0.453** | **0.723** | **0.557** | **0.584** |
| **LOF (optimal threshold)** | **0.676** | **0.635** | **0.655** ⭐ | — |

**Winner: LOF** — roughly **2× better than Isolation Forest** across all metrics.

---

## 💼 Business Operating Points

| Scenario | Precision | Recall | Flagged | TP | FP | Use Case |
|----------|-----------|--------|---------|-----|-----|----------|
| Conservative | 0.802 | 0.466 | 86 | 69 | 17 | Auto-block high-value |
| **Balanced** ⭐ | **0.676** | **0.635** | **139** | **94** | **45** | Manual review |
| Aggressive | 0.029 | 0.851 | 4,273 | 126 | 4,147 | Secondary filter |

**Recommended deployment:** LOF at balanced threshold — catches 64% of fraud with only 45 false alarms per 85K transactions.

---

## 📁 Project Structure

```
fraud-detection/
├── data/                          # Dataset (not committed)
├── notebooks/
│   └── fraud_detection_complete.ipynb   # Main deliverable
├── src/
│   ├── preprocessing.py           # Data loading, EDA, scaling
│   ├── models.py                  # IF + LOF wrappers
│   ├── evaluation.py              # Metrics, plots
│   ├── step2_isolation_forest.py
│   ├── step3_lof.py
│   ├── step4_comparison.py
│   └── step5_final_report.py
├── outputs/
│   ├── plots/                     # Visualizations
│   ├── reports/                   # Performance report
│   └── models/                    # Saved model
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Place `creditcard.csv` in `data/` (from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

### 5. Run the pipeline
```bash
python src/preprocessing.py
python src/step2_isolation_forest.py
python src/step3_lof.py
python src/step4_comparison.py
python src/step5_final_report.py
```

Or open `notebooks/fraud_detection_complete.ipynb` for the full walkthrough.

---

## 📈 Visualizations

- **EDA** — class imbalance, amount distribution, time distribution
- **Confusion matrices** — for both IF and LOF
- **Precision-Recall curves** — side-by-side comparison
- **PCA scatter plots** — where anomalies sit in feature space
- **Score distributions** — how well models separate fraud from normal

---

## 🔮 Future Work

- **Feature engineering** — velocity, hour-of-day, merchant category
- **Ensemble** — combine IF + LOF scores
- **Deep learning** — autoencoders / LSTM for sequence anomalies
- **Concept drift** — online retraining pipeline
- **Production API** — FastAPI service with real-time scoring

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Data:** Pandas, NumPy
- **ML:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Notebook:** Jupyter

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Govind**  
[GitHub](https://github.com/mrgovind111) • [Email](mailto:rajugovind3722@gmail.com)

---

⭐ If this project helped you, consider giving it a star!