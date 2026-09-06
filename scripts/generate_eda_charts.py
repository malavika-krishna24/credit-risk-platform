"""
scripts/generate_eda_charts.py

Generates all EDA charts used in notebooks/eda.ipynb and ui/app.py's EDA
section. Run this any time the underlying data changes and charts need
regenerating.

Usage:
    python scripts/generate_eda_charts.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"

OUT_DIR = "notebooks/eda_charts"


def main():
    df = pd.read_parquet("data/features_train.parquet")

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(6, 4.3))
    counts = df["TARGET"].value_counts()
    ax.bar(["Repaid (0)", "Default (1)"], counts.values, color=["#4C9A6B", "#C0392B"])
    ax.set_title("Target Class Distribution \u2014 Severe Imbalance (~8% default)", pad=18)
    ax.set_ylabel("Applicant Count")
    ax.set_ylim(0, counts.values.max() * 1.18)  # headroom so labels never collide with the title
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.values.max() * 0.025, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_target_distribution.png", dpi=110)
    plt.close()

    # 2. Default rate by age group
    d = df.copy()
    d["age_group"] = pd.cut(d["YEARS_BIRTH"], bins=[20, 30, 40, 50, 60, 70],
                             labels=["20-30", "30-40", "40-50", "50-60", "60-70"])
    grp = d.groupby("age_group", observed=True)["TARGET"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(grp.index.astype(str), grp.values, color="#2E86AB")
    ax.set_title("Default Rate by Age Group \u2014 Younger Applicants Are Riskier", pad=14)
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Age Group")
    ax.set_ylim(0, grp.values.max() * 1.2)
    for i, v in enumerate(grp.values):
        ax.text(i, v + grp.values.max() * 0.03, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_default_by_age.png", dpi=110)
    plt.close()

    # 3. Default rate by income type
    grp = df.groupby("NAME_INCOME_TYPE")["TARGET"].agg(["mean", "count"])
    grp = grp[grp["count"] > 50].sort_values("mean", ascending=False) * [100, 1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(grp.index.astype(str), grp["mean"], color="#E67E22")
    ax.set_title("Default Rate by Employment/Income Type", pad=14)
    ax.set_xlabel("Default Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_default_by_income_type.png", dpi=110)
    plt.close()

    # 4. Bureau overdue history vs default
    d = df.copy()
    d["bureau_overdue"] = np.where(d["bureau_overdue_loan_count"].fillna(0) > 0,
                                     "Has Overdue\nBureau Loan", "No Overdue\nBureau Loan")
    grp = d.groupby("bureau_overdue")["TARGET"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.bar(grp.index, grp.values, color=["#C0392B", "#4C9A6B"])
    ax.set_title("External Bureau History Is a Strong Risk Signal\n(2x default rate with overdue bureau loans)", pad=14)
    ax.set_ylabel("Default Rate (%)")
    ax.set_ylim(0, grp.values.max() * 1.2)
    for i, v in enumerate(grp.values):
        ax.text(i, v + grp.values.max() * 0.03, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_default_by_bureau_overdue.png", dpi=110)
    plt.close()

    # 5. Prior refusal history vs default
    d = df.copy()
    d["refused"] = np.where(d["prev_refused_count"].fillna(0) > 0, "Previously\nRefused", "Never\nRefused")
    grp = d.groupby("refused")["TARGET"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.bar(grp.index, grp.values, color=["#C0392B", "#4C9A6B"])
    ax.set_title("Prior Loan Refusal History Predicts Future Default", pad=14)
    ax.set_ylabel("Default Rate (%)")
    ax.set_ylim(0, grp.values.max() * 1.2)
    for i, v in enumerate(grp.values):
        ax.text(i, v + grp.values.max() * 0.03, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_default_by_prior_refusal.png", dpi=110)
    plt.close()

    # 6. Credit-Income ratio distribution split by target
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target, color, label in [(0, "#4C9A6B", "Repaid"), (1, "#C0392B", "Default")]:
        subset = df[(df["TARGET"] == target) & (df["CREDIT_INCOME_RATIO"] < 15)]["CREDIT_INCOME_RATIO"]
        sns.kdeplot(subset, ax=ax, color=color, label=label, fill=True, alpha=0.3)
    ax.set_title("Credit-to-Income Ratio Distribution by Outcome", pad=14)
    ax.set_xlabel("Credit / Income Ratio")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_credit_income_ratio_dist.png", dpi=110)
    plt.close()

    # 7. Missing data overview (top 20)
    missing = (df.isnull().mean() * 100).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(missing.index[::-1], missing.values[::-1], color="#95A5A6")
    ax.set_title("Top 20 Columns by Missing Data %", pad=14)
    ax.set_xlabel("% Missing")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_missing_data.png", dpi=110)
    plt.close()

    print("All 7 EDA charts regenerated in", OUT_DIR)


if __name__ == "__main__":
    main()
