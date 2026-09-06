import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '..')

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
pd.set_option('display.max_columns', 50)

df = pd.read_parquet('../data/features_train.parquet')
print(f"Shape: {df.shape[0]:,} applicants x {df.shape[1]} features")
df.head()

dtype_counts = df.dtypes.value_counts()
print("Feature types:")
print(dtype_counts)

print("\nFeature groups:")
print("  Demographics/application core: ~40 columns (income, family, housing, employment)")
print("  Building/housing quality:      ~50 columns (mostly the *_AVG/_MODE/_MEDI apartment features)")
print("  Bureau (external credit):       9 engineered columns")
print("  Previous applications:          5 engineered columns")
print("  POS/Cash monthly history:       8 engineered columns")
print("  Engineered ratios:              4 columns (credit/income, annuity/income, etc.)")

missing = (df.isnull().mean() * 100).sort_values(ascending=False)
print(f"Columns with >40% missing: {(missing > 40).sum()}")
print(f"Columns with 0% missing:   {(missing == 0).sum()}")

img = plt.imread('eda_charts/07_missing_data.png')
plt.figure(figsize=(10,7))
plt.imshow(img)
plt.axis('off')
plt.show()

print(df['TARGET'].value_counts())
print(df['TARGET'].value_counts(normalize=True) * 100)

img = plt.imread('eda_charts/01_target_distribution.png')
plt.figure(figsize=(6,4))
plt.imshow(img)
plt.axis('off')
plt.show()

img = plt.imread('eda_charts/02_default_by_age.png')
plt.figure(figsize=(7,4))
plt.imshow(img)
plt.axis('off')
plt.show()

img = plt.imread('eda_charts/03_default_by_income_type.png')
plt.figure(figsize=(8,4.5))
plt.imshow(img)
plt.axis('off')
plt.show()

img = plt.imread('eda_charts/04_default_by_bureau_overdue.png')
plt.figure(figsize=(6,4))
plt.imshow(img)
plt.axis('off')
plt.show()

img = plt.imread('eda_charts/05_default_by_prior_refusal.png')
plt.figure(figsize=(6,4))
plt.imshow(img)
plt.axis('off')
plt.show()

img = plt.imread('eda_charts/06_credit_income_ratio_dist.png')
plt.figure(figsize=(7,4.5))
plt.imshow(img)
plt.axis('off')
plt.show()
