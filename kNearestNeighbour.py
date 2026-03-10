#%%
#-----------------------------------
# 0. Imports
#-----------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve, cross_val_score, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score, roc_curve, precision_recall_curve, average_precision_score)
from sklearn import model_selection, neighbors, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 

#%%
#-----------------------------------
# 1. Load data
#-----------------------------------

from ecg.load_data import load_data
data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data = pd.read_csv('ecg/ecg_data/ecg_data.csv',index_col=0)
df = pd.read_csv('ecg/ecg_data/ecg_data.csv', index_col=0)


# Drop index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "label" not in df.columns:
    raise ValueError("Expected a column named 'label' in the CSV.")

X = df.drop(columns=["label"])
y = df["label"].astype(int)

print("\nDataset shape:", X.shape)
print("Label distribution:\n", y.value_counts())

#%%
#-----------------------------------
# 2. Train/Test split (hold-out)
#-----------------------------------   
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    shuffle=True,
    stratify=y,
    random_state=42
)

print("\nTrain label distribution:\n", y_train.value_counts())
print("Test  label distribution:\n", y_test.value_counts())
