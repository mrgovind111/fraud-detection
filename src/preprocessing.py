import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_data(path='data/creditcard.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found at " + path)

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)
    print("Shape:            " + str(df.shape))
    print("Total txns:       " + str(len(df)))
    print("Fraud txns:       " + str(df['Class'].sum()))
    print("Normal txns:      " + str((df['Class'] == 0).sum()))
    print("Fraud pct:        " + str(round(df['Class'].mean() * 100, 4)) + "%")
    print("Missing values:   " + str(df.isnull().sum().sum()))
    print("=" * 60)
    return df


def explore_data(df):
    os.makedirs('outputs/plots', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    counts = df['Class'].value_counts().sort_index()
    bars = axes[0].bar(['Normal', 'Fraud'], counts.values,
                       color=['steelblue', 'crimson'])
    axes[0].set_yscale('log')
    axes[0].set_title('Class Distribution (log scale)')
    axes[0].set_ylabel('Count (log)')
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     v, str(v), ha='center', va='bottom', fontsize=9)

    axes[1].hist(df['Amount'], bins=50, color='steelblue', edgecolor='white')
    axes[1].set_title('Transaction Amount Distribution')
    axes[1].set_xlabel('Amount')
    axes[1].set_ylabel('Frequency')
    axes[1].set_yscale('log')

    axes[2].hist(df['Time'], bins=50, color='seagreen', edgecolor='white')
    axes[2].set_title('Time Distribution')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('outputs/plots/01_eda_overview.png', bbox_inches='tight')
    plt.close()
    print("Saved: outputs/plots/01_eda_overview.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    data_to_plot = [df.loc[df['Class'] == 0, 'Amount'],
                    df.loc[df['Class'] == 1, 'Amount']]
    ax.boxplot(data_to_plot)
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yscale('log')
    ax.set_title('Transaction Amount by Class (log scale)')
    ax.set_ylabel('Amount (log)')
    plt.tight_layout()
    plt.savefig('outputs/plots/02_amount_by_class.png', bbox_inches='tight')
    plt.close()
    print("Saved: outputs/plots/02_amount_by_class.png")

    print("")
    print("--- Amount Statistics ---")
    print(df.groupby('Class')['Amount'].describe().round(2))
    print("")
    print("--- Time Statistics ---")
    print(df.groupby('Class')['Time'].describe().round(2))


def preprocess(df):
    df = df.copy()
    scaler = RobustScaler()
    df[['Amount_scaled', 'Time_scaled']] = scaler.fit_transform(
        df[['Amount', 'Time']]
    )
    df = df.drop(['Amount', 'Time'], axis=1)

    cols = [c for c in df.columns if c != 'Class'] + ['Class']
    df = df[cols]

    print("")
    print("Preprocessing complete.")
    print("Final shape: " + str(df.shape))
    return df, scaler


if __name__ == '__main__':
    df = load_data('data/creditcard.csv')
    explore_data(df)
    df_scaled, scaler = preprocess(df)

    os.makedirs('data', exist_ok=True)
    df_scaled.to_csv('data/creditcard_scaled.csv', index=False)
    print("")
    print("Saved scaled data: data/creditcard_scaled.csv")
    print("")
    print("[STEP 1 COMPLETE]")