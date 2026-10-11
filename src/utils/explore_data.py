"""
Dataset Exploration Script
Loads UCI #697 and generates basic statistics + plots.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (saves plots as files)
import matplotlib.pyplot as plt
import seaborn as sns


def load_uci697():
    """Load the main training dataset."""
    path = 'data/raw/uci_697_dropout.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, sep=';')


def main():
    print("=" * 60)
    print("DATASET EXPLORATION — UCI #697")
    print("=" * 60)

    df = load_uci697()

    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"\nTarget distribution:\n{df['Target'].value_counts()}")
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"\nData types count:\n{df.dtypes.value_counts()}")

    # Ensure figures folder exists
    os.makedirs('reports/figures', exist_ok=True)

    # Plot 1: Target distribution
    plt.figure(figsize=(8, 5))
    colors = {'Graduate': '#27ae60', 'Dropout': '#e74c3c', 'Enrolled': '#f39c12'}
    counts = df['Target'].value_counts()
    bar_colors = [colors[label] for label in counts.index]
    counts.plot(kind='bar', color=bar_colors)
    plt.title('Target Class Distribution — UCI #697')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/figures/target_distribution.png', dpi=100)
    plt.close()
    print("\n✓ Saved: reports/figures/target_distribution.png")

    # Plot 2: Correlation heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap — UCI #697')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png', dpi=100)
    plt.close()
    print("✓ Saved: reports/figures/correlation_heatmap.png")

    print("\n" + "=" * 60)
    print("✓ EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()