# ## Data loading and cleaning
#%%
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

# %% SPLITTEN VAN DE DATA
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)


# %% HET BEPALEN VAN CUMULATIEVE VARIANTIE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
import pylab as pl
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
print("The shape of Feature Matrix is -",X_std.shape)
X_covariance_matrix = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(X_covariance_matrix)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Variance captured by each component is \n",var_exp)
print(40 * '-')
print("Cumulative variance captured as we travel each component \n",cum_var_exp)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

#%% GENERALIZATION
from sklearn.decomposition import PCA 

# PCA fitten **alleen op trainingsdata**
pca = PCA(n_components=0.95)                        # behoudt 95% van de variantie
X_train_pca = pca.fit_transform(X_train)

# testdata transformeren met dezelfde PCA
X_test_pca = pca.transform(X_test)

