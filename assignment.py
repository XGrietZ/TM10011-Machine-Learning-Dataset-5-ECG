# ## Data loading and cleaning
from ecg.load_data import load_data
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
import numpy as np

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data = pd.read_csv('ecg\ecg_data\ecg_data.csv',index_col=0)

# print(sum(data.iloc[:,-1:]))
# print(data['label'].sum())

# print(data.iloc[750])
# Er zijn 146 gezonde patienten en 691 ongezonde patienten

# %% SPLITTEN VAN DE DATA
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, stratify=y)
