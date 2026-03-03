# Margriets code for loading and exploring the ECG dataset.

# %% Import libraries & load dataset
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import torch
import pandas as pd    
import numpy as np
import seaborn as sns

from scipy.signal import savgol_filter

from ecg.load_data import load_data

# Load the dataset
data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')



#%% Prep data
X = data.iloc[:, :-1]                           # shape (n_patients, 9000), laatste kolom is label

n_leads = 12                                    # Number of leads in the ECG data
features_per_lead = 750                         # Number of features per lead (9000 total / 12 leads)
fs = 500                                        # samplefrequentie in Hz

patient_index = 0
patient = X.iloc[patient_index].values          # shape (9000,)

patient_reshaped = patient.reshape(n_leads, features_per_lead)  # Reshape naar 12 leads

freq = np.linspace(0, fs/2, features_per_lead)  # Frequency axis

# band_mask = (freq >= 0) & (freq <= 40)          # Filteren

#%% Figure 
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
axes = axes.flatten()

for i in range(n_leads):
    lead = patient_reshaped[i]
    
    # # Filtering in frequency domain
    # filtered = lead[band_mask]
    # filtered_freq = freq[band_mask]

        
    # Plot
    axes[i].plot(freq, lead)
    axes[i].set_title(f"Lead {i+1}")
    axes[i].set_xlabel("Frequency (Hz)")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid(True)

plt.tight_layout()
plt.show()
# %%
