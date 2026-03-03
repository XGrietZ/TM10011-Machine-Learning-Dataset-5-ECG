#%% Import libraries & load dataset
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import torch
import pandas as pd    
import numpy as np
import seaborn as sns

from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data = pd.DataFrame(data)

#%% Split Data in 80% training and 20% test set
from sklearn.model_selection import train_test_split
X = data.iloc[:, :-1]                           # shape (n_patients, 9000), last column contains label
y = data.iloc[:, -1]                            # shape (n_patients, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print(f'The number of samples in the training set: {len(X_train)}')
print(f'The number of samples in the test set: {len(X_test)}')
