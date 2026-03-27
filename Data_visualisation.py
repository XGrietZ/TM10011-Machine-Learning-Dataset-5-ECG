#%%
# -----------------------------------
# 0. Imports
# -----------------------------------
from ecg.load_data import load_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%%
# -----------------------------------
# 1. Load and preprocess data
# -----------------------------------
data = load_data()

print(f"Number of samples: {len(data.index)}")
print(f"Number of columns: {len(data.columns)}")

# Convert to DataFrame for consistency
df = pd.DataFrame(data)

# Remove unwanted index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure label column exists
if "label" not in df.columns:
    raise ValueError("Expected a column named 'label' in the dataset.")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"].astype(int)

print("\nDataset shape:", X.shape)
print("Label distribution:\n", y.value_counts())

#%%
# -----------------------------------
# 2. Frequency-domain visualization
# -----------------------------------

# Sampling parameters
fs = 500                      # Sampling frequency (Hz)
n_bins = 750                  # Frequency bins per lead
freqs = np.linspace(0, fs/2, n_bins)

# ECG lead names
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Frequency range of interest
freq_min = 0
freq_max = 100
freq_mask = (freqs >= freq_min) & (freqs <= freq_max)

# Number of features per lead in selected range
n_features_per_lead = np.sum(freq_mask)
print(f"\nFeatures per lead in {freq_min}-{freq_max} Hz: {n_features_per_lead}")

# Select one example spectrum
sample_idx = 0

# Plot all 12 leads
fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle("Frequency-domain representation of all 12 ECG leads", fontsize=14)

for lead in range(12):
    row = lead // 3
    col = lead % 3
    ax = axes[row, col]

    # Extract spectrum for current lead
    start = lead * n_bins
    end = (lead + 1) * n_bins
    spectrum = X.iloc[sample_idx, start:end].values

    ax.plot(freqs[freq_mask], spectrum[freq_mask], linewidth=1)
    ax.set_title(lead_names[lead], fontsize=10)

    if row == 3:
        ax.set_xlabel("Frequency (Hz)")
    if col == 0:
        ax.set_ylabel("Amplitude")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%
# -----------------------------------
# 3. Train / test split
# -----------------------------------
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    shuffle=True,
    random_state=42
)

print("\nTrain set label distribution:\n", y_train_full.value_counts())
print("Test set label distribution:\n", y_test_final.value_counts())

#%%
# -----------------------------------
# 4. Explained variance analysis
# -----------------------------------

# Fit PCA on standardized training data
pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# Define variance threshold
threshold = 0.935
idx_94 = np.argmax(cum_var >= threshold)

plt.figure(figsize=(8, 5))

# Plot variance up to threshold (blue region)
plt.plot(
    range(idx_94 + 1),
    cum_var[:idx_94 + 1],
    label="≤ 93,5% variance"
)

# Plot variance beyond threshold (red region)
plt.plot(
    range(idx_94, len(cum_var)),
    cum_var[idx_94:],
    label="> 93,5% variance"
)

# Threshold line
plt.axhline(threshold, linestyle='--', label="93,5% threshold")

# Vertical line at cutoff
plt.axvline(idx_94, linestyle=':', alpha=0.7)

# Annotation of selected number of components
plt.text(idx_94 + 5, threshold - 0.04, f"{idx_94} components", fontsize=10)

# Labels and title
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance")

plt.legend()
plt.tight_layout()
plt.show()
# %%
