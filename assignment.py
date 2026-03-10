#%% DATA LOADING AND IMPORTING PACKAGES

# Importing packages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
import numpy as np

# Loading the data
from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data = pd.read_csv('ecg\ecg_data\ecg_data.csv',index_col=0)

# print(sum(data.iloc[:,-1:]))
# print(data['label'].sum())

# %% SPLITTEN VAN DE DATA
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

#%% GENERALIZATION
from sklearn.decomposition import PCA 

# PCA fitten **alleen op trainingsdata**
pca = PCA(n_components=0.95)                        # behoudt 95% van de variantie
X_train_pca = pca.fit_transform(X_train)

# testdata transformeren met dezelfde PCA
X_test_pca = pca.transform(X_test)

