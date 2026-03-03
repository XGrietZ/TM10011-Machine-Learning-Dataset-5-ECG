#%% Import libraries & load dataset
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import torch
import pandas as pd    
import numpy as np
import seaborn as sns

from ecg.load_data import load_data
from sklearn import model_selection

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data = pd.read_csv('ecg\ecg_data\ecg_data.csv',index_col=0)

# print(sum(data.iloc[:,-1:]))
# print(data['label'].sum())

# %% SPLITTEN VAN DE DATA

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, stratify=y)

print(f'The number of samples in the training set: {len(X_train)}')
print(f'The number of samples in the test set: {len(X_test)}')
